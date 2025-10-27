(function floodDroughtPage(){
const time = document.getElementById('timeHorizon');
const timeLabel = document.getElementById('timeHorizonLabel');
const regionSelect = document.getElementById('regionSelect');
const btnApply = document.getElementById('btnApply');
const btnReset = document.getElementById('btnReset');
const annotations = document.getElementById('annotations');


if (time && timeLabel) {
time.addEventListener('input', () => { timeLabel.textContent = time.value; });
}


if (btnApply && annotations) {
btnApply.addEventListener('click', () => {
const selected = regionSelect?.value || '—';
const layers = [
['Flood risk', document.getElementById('layerFlood')?.checked],
['Drought risk', document.getElementById('layerDrought')?.checked],
['Affordability', document.getElementById('layerAfford')?.checked],
['Incidents', document.getElementById('layerIncidents')?.checked],
].filter(([, on]) => on).map(([name]) => name).join(', ');


annotations.innerHTML = `
<div class="card">
<h3>Applied</h3>
<p><strong>Region:</strong> ${selected}</p>
<p><strong>Layers:</strong> ${layers || 'None'}</p>
<p><strong>Horizon:</strong> ${time?.value || '—'} months</p>
<p class="muted">(Data and map overlays will render here in a later sprint.)</p>
</div>`;
});
}


if (btnReset && annotations) {
btnReset.addEventListener('click', () => {
if (regionSelect) regionSelect.value = '';
['layerFlood','layerDrought','layerAfford','layerIncidents'].forEach(id => {
const el = document.getElementById(id);
if (el && (id === 'layerFlood' || id === 'layerDrought')) el.checked = true; else if (el) el.checked = false;
});
if (time) { time.value = 6; document.getElementById('timeHorizonLabel').textContent = '6'; }
annotations.innerHTML = '<p class="muted">Annotations will appear here.</p>';
});
}
})();