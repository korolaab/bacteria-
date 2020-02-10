package main
import (
	"net/http"
	"net/http/fcgi"
	"fmt")

func main(){
    listener,e:=net.Listen("tcp","127.0.0.1:9991")
    if e != nil{
    panic(e)
    }
    h1 := func(w http.ResponseWriter, _ *http.Request){
        b,err := ioutil.ReadFile("web/index.html")
        if err != nil{
            return
            }
        fmt.Fprintf(w,string(b))
        }
    mux := http.NewServeMux()
    http.HandleFunc("/bacteria/net",inference.image_to_network) //neural network inference
    mux.HandleFunc("/bacteria",page.index) //main page with information
    mux.HandleFunc("/bacteria",page.segmentation) // segmentation web page
    fcgi.Serve(listener,nil)
}
