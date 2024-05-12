function submitForm(){
    var conf = document.querySelector('#confRange').value;
    var iou = document.querySelector('#iouRange').value;
    var selectList = document.querySelector('#image_index').value;
    var xmlHttp = new XMLHttpRequest();

    var data = {
        'conf': conf,
        'iou': iou,
        'image_index': selectList
      };
      
    var searchParams = new URLSearchParams(data);
    
    xmlHttp.open( "GET", '/ui?'+ searchParams.toString(), false ); // false for synchronous request
    xmlHttp.send( null );
    console.log(xmlHttp.responseText);
    return xmlHttp.responseText;
}