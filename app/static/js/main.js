document.querySelectorAll('form[method="post"]').forEach(form=>{if(!form.querySelector('[name="csrf_token"]')){const token=document.createElement('input');token.type='hidden';token.name='csrf_token';token.value=document.querySelector('meta[name="csrf-token"]')?.content||'';form.appendChild(token)}});document.querySelectorAll('form[data-confirm]').forEach(form=>form.addEventListener('submit',e=>{if(!confirm(form.dataset.confirm))e.preventDefault()}));
const input=document.getElementById('video');
let previewObjectUrl=null;
if(input){input.addEventListener('change',()=>{const file=input.files?.[0];if(!file)return;document.getElementById('file-name').textContent=file.name;const wrap=document.getElementById('preview-wrap');const video=document.getElementById('preview');if(previewObjectUrl)URL.revokeObjectURL(previewObjectUrl);previewObjectUrl=URL.createObjectURL(file);video.pause();video.src=previewObjectUrl;video.load();wrap.hidden=false;});const dz=document.querySelector('.dropzone');['dragenter','dragover'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.add('dragging')}));['dragleave','drop'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.remove('dragging')}));dz.addEventListener('drop',e=>{if(e.dataTransfer.files.length){input.files=e.dataTransfer.files;input.dispatchEvent(new Event('change'))}})}
const uploadForm=document.getElementById('upload-form');
if(uploadForm&&input){
  const feedback=document.getElementById('upload-feedback'),progress=document.getElementById('upload-progress'),submit=uploadForm.querySelector('[type="submit"]');
  const serverMessage=xhr=>{
    // Flask returns a safe, user-facing message in its error page. Surface that
    // message instead of replacing every server failure with the same generic text.
    try{
      const page=new DOMParser().parseFromString(xhr.responseText,'text/html');
      const message=page.querySelector('.empty p');
      if(message?.textContent.trim())return message.textContent.trim();
    }catch(_error){}
    return xhr.status===0?'The server could not be reached during upload. Try again.':`The server could not complete the upload (HTTP ${xhr.status}). Check the Flask terminal for the recorded error.`;
  };
  input.addEventListener('change',()=>{const file=input.files[0];if(file){document.getElementById('file-size').textContent=`${(file.size/1048576).toFixed(2)} MB`;}});
  uploadForm.addEventListener('submit',event=>{
    const file=input.files[0];if(!file)return;
    event.preventDefault();
    const allowed=['mp4','avi','mov','mkv'];
    const extension=file.name.split('.').pop().toLowerCase();
    if(!allowed.includes(extension)){feedback.hidden=false;feedback.textContent='Choose an MP4, AVI, MOV, or MKV video.';return;}
    if(file.size===0){feedback.hidden=false;feedback.textContent='The selected file is empty. Choose a non-empty video.';return;}
    if(file.size>Number(uploadForm.dataset.maxBytes)){feedback.hidden=false;feedback.textContent='The selected video exceeds the configured upload limit.';return;}
    submit.disabled=true;feedback.hidden=false;feedback.textContent='Uploading video…';progress.hidden=false;progress.value=0;
    const xhr=new XMLHttpRequest();xhr.open('POST',uploadForm.action);xhr.upload.addEventListener('progress',e=>{if(e.lengthComputable){progress.value=Math.round(100*e.loaded/e.total);feedback.textContent=`Uploading video… ${progress.value}%`;}});
    xhr.addEventListener('load',()=>{if(xhr.status>=200&&xhr.status<400&&xhr.responseURL){progress.value=100;feedback.textContent='Upload received. Opening the saved analysis record…';window.location.assign(xhr.responseURL);}else{submit.disabled=false;feedback.textContent=xhr.status===413?'The video exceeds the server upload limit.':serverMessage(xhr);}});
    xhr.addEventListener('error',()=>{submit.disabled=false;feedback.textContent=serverMessage(xhr);});
    xhr.send(new FormData(uploadForm));
  });
}
