#!/usr/bin/env guile
!#
;;; scheme_runner.scm -- JSON-RPC 2.0 REPL runner for SchemeInterpreter
;;;
;;; This script is spawned as a subprocess by SchemeInterpreter. It:
;;; - Reads JSON-RPC 2.0 messages (one per line) from stdin
;;; - Handles "execute": evaluates Scheme code, captures output, returns result
;;; - Handles "register": defines tool wrapper functions and SUBMIT in the environment
;;; - Handles "shutdown": exits cleanly
;;; - Sends "tool_call" JSON-RPC requests to stdout for host-side tool invocation
;;; - Maintains persistent state across execute calls
;;;
;;; Communication protocol: newline-delimited JSON on stdin/stdout.

(use-modules (ice-9 rdelim)
             (ice-9 ports)
             (ice-9 eval-string)
             (ice-9 regex)
             (ice-9 format))

(define (json-write-string str port)
  (display #\" port)
  (string-for-each
   (lambda (ch)
     (cond
      ((char=? ch #\") (display "\\\"" port))
      ((char=? ch #\\) (display "\\\\" port))
      ((char=? ch #\newline) (display "\\n" port))
      ((char=? ch #\return) (display "\\r" port))
      ((char=? ch #\tab) (display "\\t" port))
      ((char<? ch #\space)
       (format port "\\u~4,'0x" (char->integer ch)))
      (else (display ch port))))
   str)
  (display #\" port))

(define (every-pair-with-string-key? lst)
  (and (pair? lst)
       (let loop ((items lst))
         (if (null? items)
             #t
             (and (pair? (car items))
                  (or (string? (caar items)) (symbol? (caar items)))
                  (loop (cdr items)))))))

(define (json-write obj port)
  (cond
   ((eq? obj #t) (display "true" port))
   ((eq? obj #f) (display "false" port))
   ((eq? obj 'null) (display "null" port))
   ((null? obj) (display "null" port))
   ((number? obj)
    (if (and (exact? obj) (integer? obj))
        (display obj port)
        (display (exact->inexact obj) port)))
   ((string? obj) (json-write-string obj port))
   ((symbol? obj) (json-write-string (symbol->string obj) port))
   ((vector? obj)
    (display "[" port)
    (let ((len (vector-length obj)))
      (do ((i 0 (+ i 1)))
          ((= i len))
        (when (> i 0) (display "," port))
        (json-write (vector-ref obj i) port)))
    (display "]" port))
   ((and (list? obj) (not (null? obj)) (every-pair-with-string-key? obj))
    (display "{" port)
    (let loop ((items obj) (first? #t))
      (unless (null? items)
        (unless first? (display "," port))
        (let ((key (if (symbol? (caar items)) (symbol->string (caar items)) (caar items)))
              (val (cdar items)))
          (json-write-string key port)
          (display ":" port)
          (json-write val port))
        (loop (cdr items) #f)))
    (display "}" port))
   ((list? obj)
    (display "[" port)
    (let loop ((items obj) (first? #t))
      (unless (null? items)
        (unless first? (display "," port))
        (json-write (car items) port)
        (loop (cdr items) #f)))
    (display "]" port))
   ((pair? obj)
    (display "[" port)
    (json-write (car obj) port)
    (display "," port)
    (json-write (cdr obj) port)
    (display "]" port))
   (else
    (json-write-string (format #f "~a" obj) port))))

(define (json-write-to-string obj)
  (call-with-output-string (lambda (port) (json-write obj port))))

(define (json-read port)
  (json-skip-whitespace port)
  (let ((ch (peek-char port)))
    (cond
     ((eof-object? ch) (error "Unexpected EOF in JSON"))
     ((char=? ch #\{) (json-read-object port))
     ((char=? ch #\[) (json-read-array port))
     ((char=? ch #\") (json-read-string port))
     ((char=? ch #\t) (json-read-true port))
     ((char=? ch #\f) (json-read-false port))
     ((char=? ch #\n) (json-read-null port))
     ((or (char=? ch #\-) (char-numeric? ch)) (json-read-number port))
     (else (error "Unexpected character in JSON" ch)))))

(define (json-skip-whitespace port)
  (let loop ()
    (let ((ch (peek-char port)))
      (when (and (not (eof-object? ch))
                 (or (char=? ch #\space) (char=? ch #\tab)
                     (char=? ch #\newline) (char=? ch #\return)))
        (read-char port)
        (loop)))))

(define (json-high-surrogate? code)
  (and (>= code #xD800) (<= code #xDBFF)))

(define (json-low-surrogate? code)
  (and (>= code #xDC00) (<= code #xDFFF)))

(define (json-read-u16-code-unit port)
  (let* ((h1 (read-char port))
         (h2 (read-char port))
         (h3 (read-char port))
         (h4 (read-char port))
         (code (string->number (string h1 h2 h3 h4) 16)))
    (if code code (error "Invalid \\u escape sequence"))))

(define (json-read-string port)
  (read-char port)
  (let loop ((chars '()))
    (let ((ch (read-char port)))
      (cond
       ((eof-object? ch) (error "Unterminated string"))
       ((char=? ch #\") (list->string (reverse chars)))
       ((char=? ch #\\)
        (let ((esc (read-char port)))
          (cond
           ((char=? esc #\") (loop (cons #\" chars)))
           ((char=? esc #\\) (loop (cons #\\ chars)))
           ((char=? esc #\/) (loop (cons #\/ chars)))
           ((char=? esc #\n) (loop (cons #\newline chars)))
           ((char=? esc #\r) (loop (cons #\return chars)))
           ((char=? esc #\t) (loop (cons #\tab chars)))
           ((char=? esc #\b) (loop (cons #\backspace chars)))
           ((char=? esc #\f) (loop (cons #\page chars)))
           ((char=? esc #\u)
            (let ((code (json-read-u16-code-unit port)))
              (cond
               ((json-high-surrogate? code)
                (let ((slash (read-char port))
                      (u (read-char port)))
                  (if (and (char=? slash #\\) (char=? u #\u))
                      (let ((low (json-read-u16-code-unit port)))
                        (if (json-low-surrogate? low)
                            (let ((combined (+ #x10000
                                               (* (- code #xD800) #x400)
                                               (- low #xDC00))))
                              (loop (cons (integer->char combined) chars)))
                            (error "Invalid low surrogate in \\u escape")))
                      (error "Expected low surrogate after high surrogate"))))
               ((json-low-surrogate? code) (error "Unexpected low surrogate in \\u escape"))
               (else (loop (cons (integer->char code) chars))))))
           (else (loop (cons esc chars))))))
       (else (loop (cons ch chars)))))))

(define (json-read-number port)
  (let loop ((chars '()))
    (let ((ch (peek-char port)))
      (if (and (not (eof-object? ch))
               (or (char-numeric? ch)
                   (char=? ch #\-)
                   (char=? ch #\+)
                   (char=? ch #\.)
                   (char=? ch #\e)
                   (char=? ch #\E)))
          (begin
            (read-char port)
            (loop (cons ch chars)))
          (let ((s (list->string (reverse chars))))
            (or (string->number s) (error "Invalid number" s)))))))

(define (json-read-true port)
  (read-char port) (read-char port) (read-char port) (read-char port)
  #t)

(define (json-read-false port)
  (read-char port) (read-char port) (read-char port) (read-char port) (read-char port)
  #f)

(define (json-read-null port)
  (read-char port) (read-char port) (read-char port) (read-char port)
  'null)

(define (json-read-object port)
  (read-char port)
  (json-skip-whitespace port)
  (if (char=? (peek-char port) #\})
      (begin (read-char port) '())
      (let loop ((result '()))
        (json-skip-whitespace port)
        (let ((key (json-read-string port)))
          (json-skip-whitespace port)
          (read-char port)
          (json-skip-whitespace port)
          (let ((val (json-read port)))
            (let ((pair (cons key val)))
              (json-skip-whitespace port)
              (let ((ch (read-char port)))
                (cond
                 ((char=? ch #\}) (reverse (cons pair result)))
                 ((char=? ch #\,) (loop (cons pair result)))
                 (else (error "Expected , or } in object" ch))))))))))

(define (json-read-array port)
  (read-char port)
  (json-skip-whitespace port)
  (if (char=? (peek-char port) #\])
      (begin (read-char port) '())
      (let loop ((result '()))
        (json-skip-whitespace port)
        (let ((val (json-read port)))
          (json-skip-whitespace port)
          (let ((ch (read-char port)))
            (cond
             ((char=? ch #\]) (reverse (cons val result)))
             ((char=? ch #\,) (loop (cons val result)))
             (else (error "Expected , or ] in array" ch))))))))

(define (json-read-from-string str)
  (call-with-input-string str json-read))

(define (alist-ref alist key . default)
  (let ((pair (assoc key alist)))
    (if pair (cdr pair) (if (null? default) #f (car default)))))

(define (jsonrpc-result result id)
  (json-write-to-string `(("jsonrpc" . "2.0") ("result" . ,result) ("id" . ,id))))

(define (jsonrpc-error code message id . data)
  (let ((err `(("code" . ,code) ("message" . ,message))))
    (when (and (not (null? data)) (car data))
      (set! err (append err `(("data" . ,(car data))))))
    (json-write-to-string `(("jsonrpc" . "2.0") ("error" . ,err) ("id" . ,id)))))

(define (jsonrpc-request method params id)
  (json-write-to-string `(("jsonrpc" . "2.0") ("method" . ,method) ("params" . ,params) ("id" . ,id))))

(define *tool-call-id* 0)
(define *host-input-port* (current-input-port))
(define *host-output-port* (current-output-port))

(define (call-host-tool tool-name args)
  (set! *tool-call-id* (+ *tool-call-id* 1))
  (let* ((request-id (string-append "tool-" (number->string *tool-call-id*)))
         (request (jsonrpc-request "tool_call" `(("name" . ,tool-name) ("args" . ,args) ("kwargs" . ())) request-id)))
    (display request *host-output-port*)
    (newline *host-output-port*)
    (force-output *host-output-port*)
    (let* ((line (read-line *host-input-port*))
           (response (json-read-from-string line))
           (resp-result (alist-ref response "result"))
           (resp-error (alist-ref response "error")))
      (cond
       (resp-error (error (string-append "Tool error: " (alist-ref resp-error "message" "Unknown"))))
       (resp-result
        (let ((value (alist-ref resp-result "value" ""))
              (type (alist-ref resp-result "type" "string")))
          (if (string=? type "json") (json-read-from-string value) value)))
       (else (error "Invalid tool response"))))))

(define *submit-exception-key* 'dspy-submit)

(define (default-SUBMIT output)
  (throw *submit-exception-key* `(("output" . ,output))))

(define SUBMIT default-SUBMIT)

(define *sandbox-module*
  (let ((mod (make-module)))
    (set-module-uses! mod (list (resolve-interface '(guile))))
    (module-define! mod 'SUBMIT SUBMIT)
    (module-define! mod 'tool-call (lambda (name . args) (call-host-tool name args)))
    mod))

(define (sandbox-define! name value)
  (module-define! *sandbox-module* name value))

(define (sandbox-eval code-string)
  (let* ((output-str "")
         (result
          (with-output-to-string
            (lambda () (eval-string code-string #:module *sandbox-module*)))))
    result))

(define (host-respond str)
  (display str *host-output-port*)
  (newline *host-output-port*)
  (force-output *host-output-port*))

(define (handle-execute params id)
  (let ((code (alist-ref params "code" "")))
    (catch #t
      (lambda ()
        (catch *submit-exception-key*
          (lambda ()
            (let ((output (sandbox-eval code)))
              (host-respond (jsonrpc-result `(("output" . ,output)) id))))
          (lambda (key final-data)
            (host-respond (jsonrpc-result `(("final" . ,final-data)) id)))))
      (lambda (key . args)
        (let* ((error-msg
                (catch #t
                  (lambda ()
                    (cond
                     ((and (>= (length args) 3) (string? (cadr args)) (list? (caddr args)))
                      (let ((template (cadr args)) (fmt-args (caddr args)))
                        (apply format #f
                               (string-append "~a: " (regexp-substitute/global #f "~[SA]" template 'pre "~a" 'post))
                               key fmt-args)))
                     ((and (>= (length args) 2) (string? (cadr args))) (format #f "~a: ~a" key (cadr args)))
                     ((not (null? args)) (format #f "~a: ~a" key (string-join (map (lambda (a) (format #f "~a" a)) args) " ")))
                     (else (symbol->string key))))
                  (lambda _ (format #f "~a: ~a" key args))))
               (error-type (cond
                            ((eq? key 'read-error) "SyntaxError")
                            ((eq? key 'syntax-error) "SyntaxError")
                            ((eq? key 'unbound-variable) "NameError")
                            ((eq? key 'wrong-type-arg) "TypeError")
                            ((eq? key 'wrong-number-of-args) "TypeError")
                            ((eq? key 'out-of-range) "IndexError")
                            (else "RuntimeError")))
               (error-code (cond ((string=? error-type "SyntaxError") -32000) (else -32007))))
          (host-respond (jsonrpc-error error-code error-msg id `(("type" . ,error-type) ("args" . ,error-msg)))))))))

(define (handle-register params id)
  (let ((tools (alist-ref params "tools" '()))
        (outputs (alist-ref params "outputs" '())))
    (when (list? tools)
      (for-each
       (lambda (tool-info)
         (let ((name (alist-ref tool-info "name")))
           (when name
             (let ((scheme-name (string->symbol name)))
               (sandbox-define! scheme-name (lambda args (call-host-tool name args)))))))
       tools))
    (when (and (list? outputs) (not (null? outputs)))
      (let ((field-names (map (lambda (f) (alist-ref f "name")) outputs)))
        (sandbox-define!
         'SUBMIT
         (lambda args
           (if (= (length args) (length field-names))
               (let ((output-alist (map cons field-names args)))
                 (throw *submit-exception-key* output-alist))
               (error (format #f "SUBMIT expects ~a arguments (~a), got ~a"
                              (length field-names)
                              (string-join field-names ", ")
                              (length args))))))))
    (host-respond (jsonrpc-result `(("ok" . #t)) id))))

(define *running* #t)

(define (main-loop)
  (let loop ()
    (when *running*
      (let ((line (read-line *host-input-port*)))
        (when (not (eof-object? line))
          (let ((line (string-trim-both line)))
            (when (> (string-length line) 0)
              (catch #t
                (lambda ()
                  (let* ((msg (json-read-from-string line))
                         (method (alist-ref msg "method"))
                         (params (alist-ref msg "params" '()))
                         (id (alist-ref msg "id")))
                    (cond
                     ((string=? method "execute") (handle-execute params id))
                     ((string=? method "register") (handle-register params id))
                     ((string=? method "shutdown") (set! *running* #f))
                     (else (host-respond (jsonrpc-error -32601 (string-append "Unknown method: " method) id))))))
                (lambda (key . args)
                  (host-respond (jsonrpc-error -32700 (format #f "Parse error: ~a ~a" key args) 'null))))))
          (loop))))))

(main-loop)
