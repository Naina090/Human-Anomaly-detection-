const desktopButtons=[...document.querySelectorAll('.history-details-toggle')];
const mobileDetails=[...document.querySelectorAll('.history-mobile-details')];

function closeOtherDetails(except){
  desktopButtons.forEach(button=>{
    const row=document.getElementById(button.getAttribute('aria-controls'));
    if(row&&row!==except){row.hidden=true;button.setAttribute('aria-expanded','false');button.textContent='More Details';}
  });
  mobileDetails.forEach(details=>{if(details!==except)details.open=false;});
}

desktopButtons.forEach(button=>{
  button.addEventListener('click',()=>{
    const row=document.getElementById(button.getAttribute('aria-controls'));
    if(!row)return;
    const expanded=button.getAttribute('aria-expanded')==='true';
    if(!expanded)closeOtherDetails(row);
    button.setAttribute('aria-expanded',String(!expanded));
    button.textContent=expanded?'More Details':'Less Details';
    row.hidden=expanded;
  });
});

mobileDetails.forEach(details=>details.addEventListener('toggle',()=>{
  if(details.open)closeOtherDetails(details);
}));
