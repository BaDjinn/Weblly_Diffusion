import ort from 'onnxruntime-web/webgpu';

function log(i) {console.log(i); document.getElementById('status').innerText += `\n${i}`;}

/* Otteniamo la configurazione dall'URL*/

function getConfig() {
    const query = window.location.search.substring(1);
    const config = {
        model: "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
        images: "2",
    };
    let vars = query.split("&");
    
}