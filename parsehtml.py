import html
import re

def RemoveHTMLTags(document):
    unescapedDocument = html.unescape(document)
    return re.sub('<[^<]+?>', '', unescapedDocument.lower())
    
def RemoveUrl(document):
    return re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S', '', document.lower(), flags=re.MULTILINE)

documentWithHtmlTags = "&lt;A HREF=""http://www.reuters.co.uk/financeQuoteLookup.jhtml?ticker=6753.T qtype=sym infotype=info qcat=news""&gt;6753.T&lt;/A&gt;"
documentWithUrl = "USATODAY.com - The federal government "

documentWithoutHtmlTags = RemoveHTMLTags(documentWithHtmlTags)
print(documentWithoutHtmlTags)

documentWithoutUrls = RemoveUrl(documentWithUrl)
print(documentWithoutUrls)