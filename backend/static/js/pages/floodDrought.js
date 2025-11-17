(function floodDroughtPage(){
const time = document.getElementById('timeHorizon');
const timeLabel = document.getElementById('timeHorizonLabel');
const regionSelect = document.getElementById('regionSelect');
const btnApply = document.getElementById('btnApply');
const btnReset = document.getElementById('btnReset');
const annotations = document.getElementById('annotations');

// Dummy flood/drought data
const floodDroughtData = [
{
id: 1,
type: "flood",
name: "Georgia Coastal Flood Risk",
coordinates: { lat: 31.1891, lng: -81.4979 },
predictedDate: "2024-04-15",
predictedTime: "08:00 UTC",
probability: 89,
severity: "high",
affectedArea: "Savannah and surrounding coastal areas",
estimatedImpact: "15,000 residents",
precipitationForecast: "6-8 inches in 24 hours",
riverLevel: "12.5 ft (flood stage: 10 ft)",
windSpeed: "18 knots SE",
stormSurge: "4-6 feet above normal",
evacuationStatus: "Recommended for low-lying areas",
estimatedDuration: "3-5 days"
},
{
id: 2,
type: "flood",
name: "Atlanta Flash Flood Warning",
coordinates: { lat: 33.749, lng: -84.388 },
predictedDate: "2024-04-10",
predictedTime: "15:30 UTC",
probability: 76,
severity: "medium",
affectedArea: "Metropolitan Atlanta, especially Buckhead and Midtown",
estimatedImpact: "50,000 residents",
precipitationForecast: "4-5 inches in 12 hours",
riverLevel: "8.2 ft (flood stage: 7 ft)",
windSpeed: "12 knots W",
stormSurge: "N/A",
evacuationStatus: "None, shelter in place",
estimatedDuration: "1-2 days"
},
{
id: 3,
type: "drought",
name: "South Georgia Drought Alert",
coordinates: { lat: 31.5785, lng: -84.1557 },
predictedDate: "2024-06-01",
predictedTime: "12:00 UTC",
probability: 92,
severity: "high",
affectedArea: "Albany and surrounding agricultural region",
estimatedImpact: "Agricultural sector, 200,000 acres",
precipitationDeficit: "8 inches below normal (last 90 days)",
soilMoisture: "15% (critical: 20%)",
reservoirLevel: "42% capacity",
cropImpact: "Corn and cotton yields expected down 30%",
waterRestrictions: "Stage 3 in effect",
estimatedDuration: "2-4 months"
},
{
id: 4,
type: "flood",
name: "Chattahoochee River Overflow",
coordinates: { lat: 34.8895, lng: -85.2547 },
predictedDate: "2024-04-20",
predictedTime: "06:45 UTC",
probability: 68,
severity: "medium",
affectedArea: "Northwest Georgia, near Tennessee border",
estimatedImpact: "8,000 residents",
precipitationForecast: "3-4 inches in 18 hours",
riverLevel: "15.8 ft (flood stage: 15 ft)",
windSpeed: "8 knots N",
stormSurge: "N/A",
evacuationStatus: "Monitor situation",
estimatedDuration: "2-3 days"
},
{
id: 5,
type: "drought",
name: "North Georgia Water Shortage",
coordinates: { lat: 34.7465, lng: -83.9712 },
predictedDate: "2024-07-15",
predictedTime: "10:00 UTC",
probability: 81,
severity: "medium",
affectedArea: "Gainesville and Lake Lanier region",
estimatedImpact: "500,000 residents",
precipitationDeficit: "5 inches below normal (last 60 days)",
soilMoisture: "25% (critical: 20%)",
reservoirLevel: "58% capacity (Lake Lanier)",
cropImpact: "Minor impacts expected",
waterRestrictions: "Stage 2 in effect",
estimatedDuration: "6-8 weeks"
}
];

// Create modal HTML and append to body
function createModal() {
const modalHTML = `
<div id="floodModal" class="modal-overlay">
<div class="modal">
<div class="modal-header">
<h2>Flood/Drought Prediction Details</h2>
<button class="modal-close" aria-label="Close modal">&times;</button>
</div>
<div class="modal-body" id="modalContent">
<!-- Content will be dynamically inserted here -->
</div>
</div>
</div>
`;
document.body.insertAdjacentHTML('beforeend', modalHTML);

// Add event listeners for modal
const modal = document.getElementById('floodModal');
const closeBtn = modal.querySelector('.modal-close');
closeBtn.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => {
if (e.target === modal) closeModal();
});

// Close on Escape key
document.addEventListener('keydown', (e) => {
if (e.key === 'Escape' && modal.classList.contains('active')) {
closeModal();
}
});
}

function openModal(predictionData) {
const modal = document.getElementById('floodModal');
const modalContent = document.getElementById('modalContent');

const severityClass = predictionData.severity === 'high' ? 'danger' :
predictionData.severity === 'medium' ? 'warning' : 'highlight';

const probabilityClass = predictionData.probability >= 80 ? 'danger' :
predictionData.probability >= 60 ? 'warning' : 'highlight';

if (predictionData.type === 'flood') {
modalContent.innerHTML = `
<div class="modal-section">
<h3>Location</h3>
<div class="modal-info-item">
<label>Prediction Name</label>
<div class="value">${predictionData.name}</div>
</div>
<div class="modal-info-item">
<label>Coordinates</label>
<div class="value">${predictionData.coordinates.lat.toFixed(4)}°N, ${Math.abs(predictionData.coordinates.lng).toFixed(4)}°W</div>
<div class="modal-map-coords">
Lat: ${predictionData.coordinates.lat} | Lng: ${predictionData.coordinates.lng}
</div>
</div>
<div class="modal-info-item">
<label>Affected Area</label>
<div class="value">${predictionData.affectedArea}</div>
</div>
</div>

<div class="modal-section">
<h3>Prediction Details</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Predicted Date</label>
<div class="value">${new Date(predictionData.predictedDate).toLocaleDateString('en-US', {
year: 'numeric',
month: 'long',
day: 'numeric'
})}</div>
</div>
<div class="modal-info-item">
<label>Time</label>
<div class="value">${predictionData.predictedTime}</div>
</div>
<div class="modal-info-item">
<label>Probability</label>
<div class="value ${probabilityClass}">${predictionData.probability}%</div>
</div>
<div class="modal-info-item">
<label>Severity</label>
<div class="value ${severityClass}">${predictionData.severity.toUpperCase()}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Impact Assessment</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Estimated Impact</label>
<div class="value ${severityClass}">${predictionData.estimatedImpact}</div>
</div>
<div class="modal-info-item">
<label>Precipitation Forecast</label>
<div class="value">${predictionData.precipitationForecast}</div>
</div>
<div class="modal-info-item">
<label>River Level</label>
<div class="value ${severityClass}">${predictionData.riverLevel}</div>
</div>
<div class="modal-info-item">
<label>Storm Surge</label>
<div class="value">${predictionData.stormSurge}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Environmental Conditions</h3>
<div class="modal-info-item">
<label>Wind Speed & Direction</label>
<div class="value">${predictionData.windSpeed}</div>
</div>
</div>

<div class="modal-section">
<h3>Response Status</h3>
<div class="modal-info-item">
<label>Evacuation Status</label>
<div class="value">${predictionData.evacuationStatus}</div>
</div>
<div class="modal-info-item">
<label>Estimated Duration</label>
<div class="value">${predictionData.estimatedDuration}</div>
</div>
</div>
`;
} else {
// Drought modal content
modalContent.innerHTML = `
<div class="modal-section">
<h3>Location</h3>
<div class="modal-info-item">
<label>Prediction Name</label>
<div class="value">${predictionData.name}</div>
</div>
<div class="modal-info-item">
<label>Coordinates</label>
<div class="value">${predictionData.coordinates.lat.toFixed(4)}°N, ${Math.abs(predictionData.coordinates.lng).toFixed(4)}°W</div>
<div class="modal-map-coords">
Lat: ${predictionData.coordinates.lat} | Lng: ${predictionData.coordinates.lng}
</div>
</div>
<div class="modal-info-item">
<label>Affected Area</label>
<div class="value">${predictionData.affectedArea}</div>
</div>
</div>

<div class="modal-section">
<h3>Prediction Details</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Predicted Start Date</label>
<div class="value">${new Date(predictionData.predictedDate).toLocaleDateString('en-US', {
year: 'numeric',
month: 'long',
day: 'numeric'
})}</div>
</div>
<div class="modal-info-item">
<label>Time</label>
<div class="value">${predictionData.predictedTime}</div>
</div>
<div class="modal-info-item">
<label>Probability</label>
<div class="value ${probabilityClass}">${predictionData.probability}%</div>
</div>
<div class="modal-info-item">
<label>Severity</label>
<div class="value ${severityClass}">${predictionData.severity.toUpperCase()}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Drought Metrics</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Estimated Impact</label>
<div class="value ${severityClass}">${predictionData.estimatedImpact}</div>
</div>
<div class="modal-info-item">
<label>Precipitation Deficit</label>
<div class="value ${severityClass}">${predictionData.precipitationDeficit}</div>
</div>
<div class="modal-info-item">
<label>Soil Moisture</label>
<div class="value ${severityClass}">${predictionData.soilMoisture}</div>
</div>
<div class="modal-info-item">
<label>Reservoir Level</label>
<div class="value ${severityClass}">${predictionData.reservoirLevel}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Agricultural Impact</h3>
<div class="modal-info-item">
<label>Crop Impact</label>
<div class="value">${predictionData.cropImpact}</div>
</div>
</div>

<div class="modal-section">
<h3>Response Status</h3>
<div class="modal-info-item">
<label>Water Restrictions</label>
<div class="value">${predictionData.waterRestrictions}</div>
</div>
<div class="modal-info-item">
<label>Estimated Duration</label>
<div class="value">${predictionData.estimatedDuration}</div>
</div>
</div>
`;
}

modal.classList.add('active');
}

function closeModal() {
const modal = document.getElementById('floodModal');
modal.classList.remove('active');
}

function displayPredictions() {
const floodChecked = document.getElementById('layerFlood')?.checked;
const droughtChecked = document.getElementById('layerDrought')?.checked;
const filteredData = floodDroughtData.filter(item => {
if (floodChecked && item.type === 'flood') return true;
if (droughtChecked && item.type === 'drought') return true;
return false;
});

annotations.innerHTML = '';
filteredData.forEach(prediction => {
const predictionItem = document.createElement('div');
predictionItem.className = 'detection-item';
const typeIcon = prediction.type === 'flood' ? '🌊' : '🌵';
predictionItem.innerHTML = `
<h4>${typeIcon} ${prediction.name}</h4>
<p><strong>Date:</strong> ${new Date(prediction.predictedDate).toLocaleDateString()}</p>
<p><strong>Location:</strong> ${prediction.coordinates.lat.toFixed(2)}°, ${prediction.coordinates.lng.toFixed(2)}°</p>
<p><strong>Probability:</strong> ${prediction.probability}%</p>
<span class="severity ${prediction.severity}">${prediction.severity}</span>
`;
predictionItem.addEventListener('click', () => openModal(prediction));
annotations.appendChild(predictionItem);
});

if (filteredData.length === 0) {
annotations.innerHTML = '<p class="muted">No predictions match the selected layers.</p>';
}
}

if (time && timeLabel) {
time.addEventListener('input', () => { timeLabel.textContent = time.value; });
}


if (btnApply && annotations) {
btnApply.addEventListener('click', () => {
displayPredictions();
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

// Create modal on page load
createModal();
})();