<?xml version="1.0" encoding="UTF-8"?>
<!-- Based on mkdocs-rss-plugin default.xsl: adds favicon for browser tab when viewing the feed as HTML -->
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:atom="http://www.w3.org/2005/Atom"
    xmlns:dc="http://purl.org/dc/elements/1.1/">

    <xsl:output method="html" encoding="UTF-8" indent="yes"/>

    <xsl:template match="/">
        <html>
            <head>
                <meta charset="UTF-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
                <title>
                    <xsl:value-of select="rss/channel/title"/>
                </title>
                <!-- Relative to feed URL (same folder as feed_rss_created.xml); works with GitHub Pages /my-blog/ and mkdocs serve -->
                <link rel="icon" href="assets/images/favicon.png" type="image/png"/>
                <style>

*, *::before, *::after {
  box-sizing: border-box;
}

/* Anthropic 品牌浅色背景 #faf9f5，偏纸感复古 */
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  max-width: 900px;
  margin: 0 auto;
  padding: 28px 18px 40px;
  background: #faf9f5;
  color: #141413;
}

header {
  border-bottom: 1px solid #e8e6dc;
  padding-bottom: 22px;
  margin-bottom: 28px;
}

header img {
  vertical-align: middle;
  margin-right: 15px;
}
header .meta {
  margin-top: 15px;
  font-size: 0.9em;
  color: #b0aea5;
}

h1 {
  margin: 0;
  font-size: clamp(1.2rem, 5vw, 1.85rem);
  word-break: break-word;
  font-weight: 600;
  letter-spacing: -0.02em;
}

/* 每条订阅：中文宋体，西文 Times New Roman */
.item {
  background: #fffef9;
  border: 1px solid #e8e6dc;
  padding: 22px 20px;
  margin-bottom: 22px;
  border-radius: 4px;
  box-shadow: 0 1px 0 rgba(20, 20, 19, 0.06);
  font-family: "Times New Roman", Times, "SimSun", "NSimSun", "Songti SC", "STSong", "Noto Serif CJK SC", serif;
}

.item h2,
.item .meta,
.item p,
.item .categories {
  font-family: inherit;
}

.item h2 {
  margin-top: 0;
  font-size: clamp(1rem, 4vw, 1.3rem);
  word-break: break-word;
  font-weight: 600;
}

/* RSS description 常为 HTML 源码展示，保留换行便于阅读 */
.item-description {
  white-space: pre-wrap;
  word-break: break-word;
  margin: 12px 0 0;
}

.rss-copy {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.75rem;
  padding: 6px 10px;
  margin-top: 10px;
  cursor: pointer;
  color: #141413;
  background: #e8e6dc;
  border: 1px solid #b0aea5;
  border-radius: 4px;
  display: inline-block;
}

.rss-copy:hover {
  background: #faf9f5;
  border-color: #d97757;
}

.rss-copy:focus {
  outline: 2px solid #d97757;
  outline-offset: 2px;
}

.rss-copy.is-done {
  border-color: #788c5d;
  color: #3d4a2f;
}

.meta {
  font-size: 0.9em;
  color: #b0aea5;
  margin-bottom: 10px;
}

.categories span {
  background: #e8e6dc;
  border-radius: 4px;
  color: #141413;
  display: inline-block;
  font-size: 0.75em;
  margin-right: 5px;
  padding: 3px 8px;
}

.item img {
  border-radius: 6px;
  display: block;
  height: auto;
  margin: 15px 0;
  width: 100%;
  border: 1px solid #e8e6dc;
}

a {
  color: #d97757;
  text-decoration: none;
  word-break: break-word;
}

a:hover {
  text-decoration: underline;
  color: #c45f42;
}
                </style>
            </head>

            <body>

                <header>
                    <xsl:if test="rss/channel/image/url and string(rss/channel/image/url) != 'None'">
                        <img>
                            <xsl:attribute name="src">
                                <xsl:value-of select="rss/channel/image/url"/>
                            </xsl:attribute>
                            <xsl:attribute name="width">64</xsl:attribute>
                        </img>
                    </xsl:if>

                    <h1>
                        <xsl:value-of select="rss/channel/title"/>
                    </h1>

                    <p>
                        <xsl:value-of select="rss/channel/description"/>
                    </p>

                    <p>
                        <a>
                            <xsl:attribute name="href">
                                <xsl:value-of select="rss/channel/link"/>
                            </xsl:attribute>
      Visit My Blog Here!
                        </a>
                    </p>
                    <div class="meta">

                        <xsl:if test="author">
    By <xsl:value-of select="author"/>
                        </xsl:if>

                        <xsl:if test="pubDate">
    —                            <strong>Published:</strong>
                            <xsl:value-of select="pubDate"/>
                        </xsl:if>

                        <xsl:if test="updated">
    —                            <strong>Updated:</strong>
                            <xsl:value-of select="updated"/>
                        </xsl:if>

                    </div>
                </header>

                <xsl:for-each select="rss/channel/item">

                    <div class="item">

                        <h2>
                            <a>
                                <xsl:attribute name="href">
                                    <xsl:value-of select="link"/>
                                </xsl:attribute>
                                <xsl:value-of select="title"/>
                            </a>
                        </h2>

                        <div class="meta">
                            <xsl:if test="author">
      Par <xsl:value-of select="author"/>
                            </xsl:if>
                            <xsl:if test="pubDate">
      —                                <xsl:value-of select="pubDate"/>
                            </xsl:if>
                        </div>

                        <!-- Image si enclosure image/* -->
                        <xsl:if test="enclosure[starts-with(@type,'image/')]">
                            <img>
                                <xsl:attribute name="src">
                                    <xsl:value-of select="enclosure/@url"/>
                                </xsl:attribute>
                            </img>
                        </xsl:if>

                        <p class="item-description">
                            <xsl:value-of select="description"/>
                        </p>

                        <xsl:if test="normalize-space(description)">
                            <button type="button" class="rss-copy" title="复制上方正文中的 HTML 原文">
                                <xsl:attribute name="aria-label">复制本条摘要与正文中的 HTML 内容</xsl:attribute>
                                复制 HTML
                            </button>
                        </xsl:if>

                        <div class="categories">
                            <xsl:for-each select="category">
                                <span>
                                    <xsl:value-of select="."/>
                                </span>
                            </xsl:for-each>
                        </div>

                    </div>

                </xsl:for-each>

                <script>
                    <![CDATA[
(function () {
  var defaultLabel = "复制 HTML";
  document.body.addEventListener("click", function (e) {
    var btn = e.target && e.target.closest && e.target.closest(".rss-copy");
    if (!btn) return;
    var item = btn.closest && btn.closest(".item");
    var block = item && item.querySelector && item.querySelector(".item-description");
    var text = block ? (block.textContent || "") : "";
    if (!text.trim()) return;
    function ok() {
      var prev = btn.textContent;
      btn.textContent = "已复制";
      btn.classList.add("is-done");
      setTimeout(function () {
        btn.textContent = prev || defaultLabel;
        btn.classList.remove("is-done");
      }, 1600);
    }
    function fail() {
      window.prompt("请手动复制：", text);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(ok).catch(fail);
    } else {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try {
        if (document.execCommand("copy")) ok(); else fail();
      } catch (err) {
        fail();
      }
      document.body.removeChild(ta);
    }
  });
})();
                    ]]>
                </script>
            </body>
        </html>
    </xsl:template>

</xsl:stylesheet>
