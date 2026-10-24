const root=document.getElementById('analysis-result');
document.querySelectorAll('.seek-time').forEach(button=>button.addEventListener('click',()=>{const video=document.getElementById('source-video');const seconds=Number(button.dataset.seconds);if(!video||!Number.isFinite(seconds)||seconds<0)return;video.currentTime=seconds;video.play()}));
document.querySelectorAll('.result-player').forEach(player=>{
  const video=player.querySelector('video'),feedback=player.querySelector('.player-feedback'),retry=player.querySelector('.player-retry');
  if(!video||!feedback)return;
  const kind=player.dataset.playerKind||'video';
  let retryCount=0,loadingTimer=null,bufferTimer=null,stateVersion=0,lastCurrentTime=0;
  const cancelTimers=()=>{clearTimeout(loadingTimer);clearTimeout(bufferTimer)};
  const setMessage=(text,showRetry=false)=>{feedback.textContent=text;feedback.hidden=false;if(retry)retry.hidden=!showRetry};
  const markReady=()=>{stateVersion++;cancelTimers();feedback.textContent='';feedback.hidden=true;if(retry)retry.hidden=true};
  const startLoading=()=>{
    const version=++stateVersion;cancelTimers();setMessage(kind==='annotated'?'Loading annotated video…':'Loading source video…');
    loadingTimer=setTimeout(async()=>{
      if(version!==stateVersion||video.readyState>=2)return;
      let status=null,mime=null;
      try{const response=await fetch(video.currentSrc,{method:'HEAD',cache:'no-store',signal:AbortSignal.timeout(5000)});status=response.status;mime=response.headers.get('Content-Type')}
      catch(_){status=null}
      if(version!==stateVersion||video.readyState>=2)return;
      const diagnostic={player:kind,httpStatus:status,mimeType:mime,mediaErrorCode:video.error?.code||null,
        readyState:video.readyState,networkState:video.networkState,browserMessage:video.error?.message||null};
      console.warn('Sentinel video playback diagnostic',diagnostic);
      if(status===404||status===409){setMessage(status===409?'Annotated output is still processing.':'The saved video file is missing.',false);return}
      if(status&&status>=500){setMessage('The server could not provide the video. You can retry playback.',true);return}
      setMessage('The video response arrived, but this browser has not decoded a frame. Check the format or download the video.',false);
    },20000);
  };
  video.addEventListener('loadstart',startLoading);
  video.addEventListener('loadedmetadata',()=>{if(video.videoWidth&&video.videoHeight)player.style.setProperty('--video-ratio',`${video.videoWidth} / ${video.videoHeight}`)});
  ['loadeddata','canplay','playing'].forEach(name=>video.addEventListener(name,markReady));
  video.addEventListener('timeupdate',()=>{
    if(!video.paused&&video.currentTime>lastCurrentTime){lastCurrentTime=video.currentTime;markReady()}
  });
  video.addEventListener('waiting',()=>{
    if(video.error)return;
    const version=++stateVersion;clearTimeout(bufferTimer);setMessage('Video is buffering…',false);
    bufferTimer=setTimeout(()=>{if(version===stateVersion&&!video.paused&&video.readyState<3)setMessage('Still buffering. Playback will resume when more video data arrives.',false)},12000);
  });
  video.addEventListener('stalled',()=>{
    if(video.error||video.paused)return;
    const version=++stateVersion;clearTimeout(bufferTimer);setMessage('Video data is temporarily stalled; waiting for the stream to continue.',false);
    bufferTimer=setTimeout(()=>{if(version===stateVersion&&!video.paused&&video.readyState<3)setMessage('Still waiting for video data. This can happen on a slow connection.',false)},12000);
  });
  video.addEventListener('error',async()=>{
    cancelTimers();const version=++stateVersion,error=video.error;
    let status=null,mime=null,text='Video playback failed in this browser.',recoverable=false;
    if(error?.code===4)text='This video format or codec is not supported by this browser.';
    else if(error?.code===3)text='The browser decoder rejected the video data. Download the file to inspect it.';
    else if(error?.code===2){
      text='A network or server error interrupted video playback.';recoverable=true;
      try{const response=await fetch(video.currentSrc,{method:'HEAD',cache:'no-store',signal:AbortSignal.timeout(5000)});status=response.status;mime=response.headers.get('Content-Type');
        if(status===404){text=kind==='annotated'?'The annotated output file is missing. The original video remains available.':'The saved source video file is missing.';recoverable=false}
        else if(status===409){text='Annotated video is still processing.';recoverable=false}
        else if(status>=500)text='The server failed while serving this video.';
      }catch(_){status=null}
    }
    if(version!==stateVersion||video.readyState>=2)return;
    console.warn('Sentinel video playback diagnostic',{player:kind,mediaErrorCode:error?.code||null,
      browserMessage:error?.message||null,readyState:video.readyState,networkState:video.networkState,
      httpStatus:status,mimeType:mime});
    setMessage(text,recoverable&&retryCount<2);
  });
  if(video.readyState>=2)markReady();else if(video.networkState!==0)startLoading();
  retry?.addEventListener('click',()=>{
    if(retryCount>=2)return;retryCount++;stateVersion++;cancelTimers();
    setMessage(`Retrying playback (${retryCount}/2)…`,false);video.load();
  });
});
const eventRows=[...document.querySelectorAll('.timeline-event')];
const typeFilter=document.getElementById('event-type-filter');
const labelFilter=document.getElementById('event-label-filter');
const trackFilter=document.getElementById('event-track-filter');
const scoreFilter=document.getElementById('event-score-filter');
const timeFromFilter=document.getElementById('event-time-from');
const timeToFilter=document.getElementById('event-time-to');
const emptyTimeline=document.getElementById('timeline-empty');
function filterTimeline(){
  if(!eventRows.length)return;
  const type=typeFilter?.value||'',label=labelFilter?.value||'',track=trackFilter?.value||'',minimum=scoreFilter?.value===''?null:Number(scoreFilter?.value),from=timeFromFilter?.value===''?null:Number(timeFromFilter?.value),to=timeToFilter?.value===''?null:Number(timeToFilter?.value);
  let visible=0;
  eventRows.forEach(row=>{
    const score=row.dataset.score===''?null:Number(row.dataset.score);
    const time=row.dataset.time===''?null:Number(row.dataset.time);
    const show=(!type||row.dataset.type===type)&&(!label||row.dataset.label===label)&&(!track||row.dataset.track===track)&&(minimum===null||(score!==null&&score>=minimum))&&(from===null||(time!==null&&time>=from))&&(to===null||(time!==null&&time<=to));
    row.hidden=!show;if(show)visible++;
  });
  if(emptyTimeline)emptyTimeline.hidden=visible!==0;
}
[typeFilter,labelFilter,trackFilter,scoreFilter,timeFromFilter,timeToFilter].forEach(control=>control?.addEventListener('input',filterTimeline));
if(root){
  const notice=document.getElementById('processing-notice'),message=document.getElementById('processing-message'),bar=document.getElementById('analysis-progress'),badge=document.getElementById('status-badge'),faceLabel=document.getElementById('face-status-label'),poseLabel=document.getElementById('pose-status-label'),objectLabel=document.getElementById('object-status-label');
  let finished=false,requestInFlight=false,pollTimer=null,pollRevision=0,reloadScheduled=false,pollFailures=0;
  const noticeKey=`analysis-notice:${root.dataset.videoId}`;
  let pendingNotice=null;
  try{pendingNotice=sessionStorage.getItem(noticeKey);sessionStorage.removeItem(noticeKey)}catch(_){}
  const storeNotice=text=>{try{sessionStorage.setItem(noticeKey,text)}catch(_){}};
  if(pendingNotice){message.textContent=pendingNotice;notice.hidden=false;setTimeout(()=>{notice.hidden=true},5000)}
  function schedulePoll(delay){
    if(finished)return;
    clearTimeout(pollTimer);
    pollTimer=setTimeout(poll,delay);
  }
  async function poll(){
    if(finished||requestInFlight)return;
    requestInFlight=true;
    const revision=++pollRevision;
    try{
      const response=await fetch(root.dataset.statusUrl,{headers:{Accept:'application/json'},cache:'no-store'});
      if(response.status===404){
        finished=true;clearTimeout(pollTimer);pollRevision++;
        message.textContent='This video record is no longer available. Return to History and reopen a saved video.';notice.hidden=false;
        return;
      }
      const data=await response.json().catch(()=>({}));
      if(!response.ok)throw new Error(data.error||'The saved analysis status is temporarily unavailable.');
      if(finished||revision!==pollRevision)return;
      pollFailures=0;
      badge.textContent=data.status;badge.className=`badge ${data.status}`;bar.value=data.progress||0;
      if(faceLabel&&data.face_status)faceLabel.textContent=data.face_status.replaceAll('_',' ');
      if(poseLabel&&data.pose_status)poseLabel.textContent=data.pose_status.replaceAll('_',' ');
      if(objectLabel&&data.object_status)objectLabel.textContent=data.object_status.replaceAll('_',' ');
      if(data.status==='failed'||data.status==='interrupted'){
        message.textContent=data.error||'Processing did not complete.';storeNotice(`Analysis ${data.status}: ${message.textContent}`);notice.hidden=false;finished=true;clearTimeout(pollTimer);pollRevision++;
        if(!reloadScheduled){reloadScheduled=true;setTimeout(()=>location.reload(),1800)}
      }else if(data.status==='complete'){
        message.textContent='Analysis completed. Refreshing the saved results…';storeNotice('Analysis completed. Review the saved results and any candidate events.');notice.hidden=false;finished=true;clearTimeout(pollTimer);pollRevision++;
        if(!reloadScheduled){reloadScheduled=true;setTimeout(()=>location.reload(),700)}
      }else{
        notice.hidden=false;message.textContent=data.status==='queued'?'Waiting for an analysis worker…':`Processing video… ${data.progress||0}%`;schedulePoll(1400);
      }
    }catch(error){if(!finished){
      pollFailures++;
      if(pollFailures>=8){
        finished=true;clearTimeout(pollTimer);pollRevision++;
        message.textContent=`${error.message||'The saved analysis status is unavailable.'} Refresh this page to check again.`;notice.hidden=false;
      }else{
        message.textContent=`${error.message||'Cannot retrieve processing status right now.'} Retrying (${pollFailures}/8)…`;notice.hidden=false;
        schedulePoll(Math.min(3000,600*pollFailures));
      }
    }}
    finally{requestInFlight=false}
  }
  if(['queued','processing','waiting_for_mongodb'].includes(badge.textContent.trim()))schedulePoll(800);
}

const alertPanel=document.querySelector('.alert-live-status[data-alert-url]');
if(alertPanel){
  const reserved=alertPanel.querySelector('.alert-reserved-count'),overview=document.querySelector('.overview-email-status'),summary=alertPanel.querySelector('.alert-live-summary'),list=alertPanel.querySelector('.alert-live-list'),detail=alertPanel.querySelector('.alert-live-detail');
  const terminal=new Set(['sent','failed','uncertain','skipped']);
  let alertBusy=false,alertAttempts=0,alertTimer=null;
  async function refreshAlerts(){
    if(alertBusy||alertAttempts>=96)return;
    alertBusy=true;alertAttempts++;
    try{
      const response=await fetch(alertPanel.dataset.alertUrl,{headers:{Accept:'application/json'},cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const data=await response.json();
      const notifications=data.notifications||[];
      if(reserved)reserved.textContent=String(data.reserved_count||0);
      const states=notifications.map(item=>item.status==='disabled'?'skipped':(item.status||'pending'));
      const state=data.delivery_status||states[states.length-1]||'skipped';
      const title=value=>(value||'pending').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
      if(summary)summary.textContent=title(state);
      if(overview)overview.textContent=title(state);
      if(detail)detail.textContent=data.status_message||'';
      if(list){list.replaceChildren();(data.notifications||[]).forEach(item=>{const li=document.createElement('li');const categories=(item.categories||[]).map(value=>value==='SUSPECTED_FIGHTING'?'Suspected Violence-Related Activity':value.replaceAll('_',' ')).join(', ');li.textContent=`${title(item.status)}${categories?` · ${categories}`:''}${item.failure_reason?` — ${item.failure_reason}`:''}`;list.appendChild(li)})}
      if(terminal.has(state))return;
    }catch(_){if(detail)detail.textContent='Delivery status could not be refreshed; showing the last saved status and retrying.';}
    finally{alertBusy=false}
    alertTimer=setTimeout(refreshAlerts,5000);
  }
  refreshAlerts();
}
