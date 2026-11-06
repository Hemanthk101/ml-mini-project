'use strict';
const text = document.getElementById('text');
const generate = document.getElementById('generate');
const status = document.getElementById('status');
const player = document.getElementById('player');
let ready = false, active = null, submitting = false;
function message(value='', error=false) {
  status.textContent=value;
  status.hidden=!value;
  status.classList.toggle('error',error);
}
function updateButton() {
  generate.disabled=!ready || submitting || Boolean(active) || !text.value.trim();
  generate.textContent=active || submitting ? 'Generating…' : 'Generate';
}
async function api(path,payload) {
  const response=await fetch(path,payload===undefined ? {cache:'no-store'} : {
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)
  });
  const result=await response.json();
  if(!response.ok)throw new Error(result.error || 'Something went wrong. Please try again.');
  return result;
}
async function poll() {
  try {
    const state=await api('/api/status');
    ready=state.status==='ready';
    if(state.status==='error') message(state.message,true);
    else if(!ready) message('Getting ready…');
    else if(active) {
      const job=await api('/api/jobs/'+active);
      if(job.status==='completed') {
        active=null;
        player.src=job.audio;
        player.hidden=false;
        message();
      } else if(job.status==='failed') {
        active=null;
        message(job.message,true);
      } else message('Generating your speech…');
    } else if(status.textContent==='Getting ready…' || status.textContent==='Reconnecting…') message();
  } catch(error) {
    ready=false;
    message('Reconnecting…');
  }
  updateButton();
  setTimeout(poll,1500);
}
text.addEventListener('input',updateButton);
document.getElementById('form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(generate.disabled)return;
  submitting=true;
  player.pause();
  player.hidden=true;
  message('Generating your speech…');
  updateButton();
  try {
    const job=await api('/api/synthesize',{text:text.value});
    active=job.id;
  } catch(error) {message(error.message,true);}
  finally {submitting=false;updateButton();}
});
poll();
