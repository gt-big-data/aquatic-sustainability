(function oilSpillPage(){
const confidence = document.getElementById('confidence');
const confidenceLabel = document.getElementById('confidenceLabel');
const btnRunModel = document.getElementById('btnRunModel');
const detections = document.getElementById('detections');
const marineZoneSelect = document.getElementById('marineZoneSelect');


if (confidence && confidenceLabel) {
confidence.addEventListener('input', () => { confidenceLabel.textContent = confidence.value; });
}


if (btnRunModel && detections) {
btnRunModel.addEventListener('click', () => {
const zone = marineZoneSelect?.value || '—';
const now = new Date().toLocaleString();
detections.innerHTML = `
<div class="card">
<h3>Model run queued</h3>
<p><strong>Zone:</strong> ${zone}</p>
<p><strong>Confidence threshold:</strong> ${confidence?.value || '—'}%</p>
<p class="muted">${now}</p>
<p class="muted">(Spill footprints and alerts will render here in a later sprint.)</p>
</div>`;
});
}
})();