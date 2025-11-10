(function oilSpillPage(){
const confidence = document.getElementById('confidence');
const confidenceLabel = document.getElementById('confidenceLabel');
const btnRunModel = document.getElementById('btnRunModel');
const detections = document.getElementById('detections');
const marineZoneSelect = document.getElementById('marineZoneSelect');

// Dummy oil spill data
const oilSpillData = [
{
id: 1,
name: "Gulf of Mexico Incident",
coordinates: { lat: 28.7378, lng: -88.3831 },
date: "2024-03-15",
time: "14:30 UTC",
amountSpilled: "4,200 barrels",
amountLiters: "667,800 L",
cause: "Pipeline rupture",
surfaceArea: "45.2 km²",
severity: "high",
affectedArea: "Coastal waters near Louisiana",
windSpeed: "12 knots NE",
seaTemp: "22.5°C",
responseStatus: "Active cleanup in progress",
estimatedDuration: "3-4 weeks"
},
{
id: 2,
name: "North Atlantic Detection",
coordinates: { lat: 41.2345, lng: -69.8765 },
date: "2024-03-18",
time: "09:15 UTC",
amountSpilled: "850 barrels",
amountLiters: "135,150 L",
cause: "Vessel collision",
surfaceArea: "12.8 km²",
severity: "medium",
affectedArea: "Open ocean, 45 nm from shore",
windSpeed: "8 knots SW",
seaTemp: "18.2°C",
responseStatus: "Containment deployed",
estimatedDuration: "1-2 weeks"
},
{
id: 3,
name: "Caribbean Spill Event",
coordinates: { lat: 18.4789, lng: -64.5432 },
date: "2024-03-20",
time: "16:45 UTC",
amountSpilled: "2,100 barrels",
amountLiters: "333,900 L",
cause: "Tanker hull breach",
surfaceArea: "28.5 km²",
severity: "high",
affectedArea: "Near Puerto Rico coast",
windSpeed: "15 knots E",
seaTemp: "26.8°C",
responseStatus: "Emergency response active",
estimatedDuration: "2-3 weeks"
},
{
id: 4,
name: "Pacific Northwest Detection",
coordinates: { lat: 47.6234, lng: -124.3456 },
date: "2024-03-22",
time: "11:20 UTC",
amountSpilled: "320 barrels",
amountLiters: "50,880 L",
cause: "Equipment failure",
surfaceArea: "5.7 km²",
severity: "low",
affectedArea: "Washington State waters",
windSpeed: "6 knots W",
seaTemp: "12.4°C",
responseStatus: "Contained",
estimatedDuration: "3-5 days"
},
{
id: 5,
name: "Atlantic Seaboard Incident",
coordinates: { lat: 35.8976, lng: -75.2341 },
date: "2024-03-25",
time: "07:50 UTC",
amountSpilled: "1,650 barrels",
amountLiters: "262,350 L",
cause: "Illegal discharge",
surfaceArea: "18.9 km²",
severity: "medium",
affectedArea: "Off North Carolina coast",
windSpeed: "10 knots N",
seaTemp: "16.7°C",
responseStatus: "Investigation ongoing",
estimatedDuration: "1-2 weeks"
}
];

// Create modal HTML and append to body
function createModal() {
const modalHTML = `
<div id="spillModal" class="modal-overlay">
<div class="modal">
<div class="modal-header">
<h2>Oil Spill Details</h2>
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
const modal = document.getElementById('spillModal');
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

function openModal(spillData) {
const modal = document.getElementById('spillModal');
const modalContent = document.getElementById('modalContent');

const severityClass = spillData.severity === 'high' ? 'danger' :
spillData.severity === 'medium' ? 'warning' : 'highlight';

modalContent.innerHTML = `
<div class="modal-section">
<h3>Location</h3>
<div class="modal-info-item">
<label>Incident Name</label>
<div class="value">${spillData.name}</div>
</div>
<div class="modal-info-item">
<label>Coordinates</label>
<div class="value">${spillData.coordinates.lat.toFixed(4)}°N, ${Math.abs(spillData.coordinates.lng).toFixed(4)}°W</div>
<div class="modal-map-coords">
Lat: ${spillData.coordinates.lat} | Lng: ${spillData.coordinates.lng}
</div>
</div>
<div class="modal-info-item">
<label>Affected Area</label>
<div class="value">${spillData.affectedArea}</div>
</div>
</div>

<div class="modal-section">
<h3>Incident Details</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Date Occurred</label>
<div class="value">${new Date(spillData.date).toLocaleDateString('en-US', {
year: 'numeric',
month: 'long',
day: 'numeric'
})}</div>
</div>
<div class="modal-info-item">
<label>Time</label>
<div class="value">${spillData.time}</div>
</div>
<div class="modal-info-item">
<label>Severity</label>
<div class="value ${severityClass}">${spillData.severity.toUpperCase()}</div>
</div>
<div class="modal-info-item">
<label>Cause</label>
<div class="value">${spillData.cause}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Spill Metrics</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Amount Spilled</label>
<div class="value ${severityClass}">${spillData.amountSpilled}</div>
<div class="muted" style="font-size: 12px; margin-top: 4px;">(${spillData.amountLiters})</div>
</div>
<div class="modal-info-item">
<label>Surface Area</label>
<div class="value ${severityClass}">${spillData.surfaceArea}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Environmental Conditions</h3>
<div class="modal-info-grid">
<div class="modal-info-item">
<label>Wind Speed & Direction</label>
<div class="value">${spillData.windSpeed}</div>
</div>
<div class="modal-info-item">
<label>Sea Surface Temperature</label>
<div class="value">${spillData.seaTemp}</div>
</div>
</div>
</div>

<div class="modal-section">
<h3>Response Status</h3>
<div class="modal-info-item">
<label>Current Status</label>
<div class="value">${spillData.responseStatus}</div>
</div>
<div class="modal-info-item">
<label>Estimated Cleanup Duration</label>
<div class="value">${spillData.estimatedDuration}</div>
</div>
</div>
`;

modal.classList.add('active');
}

function closeModal() {
const modal = document.getElementById('spillModal');
modal.classList.remove('active');
}

function displayDetections() {
detections.innerHTML = '';
oilSpillData.forEach(spill => {
const detectionItem = document.createElement('div');
detectionItem.className = 'detection-item';
detectionItem.innerHTML = `
<h4>${spill.name}</h4>
<p><strong>Date:</strong> ${new Date(spill.date).toLocaleDateString()}</p>
<p><strong>Location:</strong> ${spill.coordinates.lat.toFixed(2)}°, ${spill.coordinates.lng.toFixed(2)}°</p>
<p><strong>Amount:</strong> ${spill.amountSpilled}</p>
<span class="severity ${spill.severity}">${spill.severity}</span>
`;
detectionItem.addEventListener('click', () => openModal(spill));
detections.appendChild(detectionItem);
});
}

// Initialize
if (confidence && confidenceLabel) {
confidence.addEventListener('input', () => {
confidenceLabel.textContent = confidence.value;
});
}

if (btnRunModel && detections) {
btnRunModel.addEventListener('click', () => {
displayDetections();
});
}

// Create modal on page load
createModal();
})();