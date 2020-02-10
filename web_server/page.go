package main
import ("net/http"
        "os"
        "io/ioutil"
        "fmt")
const  HTML_FOLDER string = "web_page/"
func make_page(title string) string{
    footer,_:=ioutil.ReadFile(HTML_FOLDER+"footer.html")
    header,_:=ioutil.ReadFile(HTML_FOLDER+"header.html")
    body,_:=ioutil.ReadFile(HTML_FOLDER+filename+".html")
    return body
    }
func get(w http.ResponseWriter, r *http.Request){
    title=r.URL.Path
    page:=make_page(title)
    w.Header().Add("Content-Type","text/html")
    fmt.Fprintf(w,page)
    return
    }

