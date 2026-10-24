const root=document.getElementById('analysis-result');
document.querySelectorAll('.seek-time').forEach(button=>button.addEventListener('click',()=>{const video=document.getElementById('source-video');const seconds=Number(button.dataset.seconds);if(!video||!Number.isFinite(seconds)||seconds<0)return;video.currentTime=seconds;video.play()}));
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
