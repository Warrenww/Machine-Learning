$('head').append(`
  <style>
    @page{
      size: A4;
      margin: calc(2.54cm / 2);
    }
    @media print{
      h1,h2,h3,p,div{font-family: Times-Roman}
      pre.part, code{background:#f7f7f7 !important}
      code.python.hljs{white-space: break-spaces;}
      .hljs-keyword, .hljs-selector-tag, .hljs-type { color: #a71d5d !important; }
      .hljs-title, .hljs-attr, .hljs-selector-id, .hljs-selector-class, .hljs-selector-attr, .hljs-selector-pseudo { color: #795da3 !important; }
      .hljs-number, .hljs-literal, .hljs-symbol, .hljs-bullet, .hljs-attribute { color: #0086b3 !important; }
      .hljs-string, .hljs-variable, .hljs-template-variable, .hljs-strong, .hljs-emphasis, .hljs-quote { color: #df5000 !important; }
      blockquote{border:none}
    }
  </style>
`)
