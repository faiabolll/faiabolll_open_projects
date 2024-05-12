var slider_conf = document.getElementById("confRange");
var output_conf = document.getElementById("confRange_value");
output_conf.innerHTML = slider_conf.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider_conf.oninput = function() {
    output_conf.innerHTML = this.value;
}

var slider_iou = document.getElementById("iouRange");
var output_iou = document.getElementById("iouRange_value");
output_iou.innerHTML = slider_iou.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider_iou.oninput = function() {
    output_iou.innerHTML = this.value;
}