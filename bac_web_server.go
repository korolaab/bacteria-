package main
import (
	"bytes"
	"compress/gzip"
	"image"
	"image/png"
	"image/gif"
	"golang.org/x/image/bmp"
	"image/jpeg"
	"net"
	"io"
	"net/http"
	"net/http/fcgi"
	"crypto/md5"
	"encoding/binary"
	"fmt")
const(MaxVarintLen64=8)
func init(){
    image.RegisterFormat("jpeg","\xff\xd8",jpeg.Decode,jpeg.DecodeConfig)
    image.RegisterFormat("png","\x89\x50\x4E\x47\x0D\x0A\x1A\x0A",png.Decode,png.DecodeConfig)
    image.RegisterFormat("gif", "\x47\x49\x46\x38\x39\x61", gif.Decode, gif.DecodeConfig)
    image.RegisterFormat("bmp", "\x42\x4D", bmp.Decode, bmp.DecodeConfig)
    }

func image_to_network(w http.ResponseWriter, r *http.Request){
    if r.Method != "POST"{
	return}
    file,_,err:= r.FormFile("image")
    if err !=nil{
	panic(err)
	}
    defer file.Close()
    //receive image from POST request
    img,_,err:=image.Decode(file)
    if err != nil{
	w.Write([]byte("Could not decode image\n"))
	return
	}
    buff :=new(bytes.Buffer)
    png.Encode(buff,img)

    input := buff.Bytes()
    //encode image as png
    var buf_img bytes.Buffer
    zw:=gzip.NewWriter(&buf_img)
    _,err=zw.Write(input)
    zw.Close()
    comp_img:=buf_img.Bytes()
    if err != nil{
	w.Write([]byte("compress error"))
	return
    }
    //compress image with gzip
    conn,err := net.Dial("tcp","127.0.0.1:9992")
    if err != nil{
	w.Write([]byte("Server error"))
	return
	}
    bytes_len :=make([]byte,8)
    check_sum_binary_len:=make([]byte,8)
    binary.BigEndian.PutUint64(bytes_len,uint64(len(comp_img)))
    check_sum := md5.Sum(comp_img)
    binary.BigEndian.PutUint64(check_sum_binary_len,uint64(len(check_sum)))
    fmt.Fprintf(conn,"%s%s%s%s",bytes_len,comp_img,check_sum_binary_len,check_sum)
    //send compressed image to python module
    //receive segmentation_map
    integer:=make([]byte,8)
    _,err=io.ReadFull(conn,integer)
    if err != nil{fmt.Fprintf(w,"recverr1:%d",8);return}
    comp_pred_len:=binary.BigEndian.Uint64(integer)
    comp_pred:=make([]byte,comp_pred_len)
    _,err=io.ReadFull(conn,comp_pred)
    if err != nil{fmt.Fprintf(w,"recverr2:%d",comp_pred_len);return}
    _,err=io.ReadFull(conn,integer)
    if err != nil{fmt.Fprintf(w,"recverr3:%d",8);return}
    check_sum_len := binary.BigEndian.Uint64(integer)
    pred_sum:=make([]byte,check_sum_len)
    _,err=io.ReadFull(conn,pred_sum)
    if err != nil{fmt.Fprintf(w,"recverr4:%d",check_sum_len);return}
    check_sum = md5.Sum(comp_pred)
    if(string(pred_sum) == string(check_sum[:])){
	r:=bytes.NewReader(comp_pred)
	zr,err:=gzip.NewReader(r)
	var resB bytes.Buffer
	if err!=nil{
	    w.Write([]byte("Error"))
	    }
	_,err=resB.ReadFrom(zr)
	zr.Close()
	seg_map:=resB.Bytes()
	fmt.Fprintf(w,"%s",seg_map)
	return
    }else{
	w.Write([]byte("archive is corrupted"))
	return
	}
    //send prediction to client
}

func main(){
    listener,e:=net.Listen("tcp","127.0.0.1:9991")
    if e != nil{
    panic(e)
    }
    http.HandleFunc("/",image_to_network)
    fcgi.Serve(listener,nil)
}
