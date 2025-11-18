/**
* Minimal map placeholder so the page has a visible canvas.
* Replace with your Maps API (Leaflet, Mapbox GL JS, Google Maps JS) later.
*/
(function initMapPlaceholder(){
const el = document.getElementById('map');
if (!el) return;


// Draw a simple crosshair & scale text to imply a map canvas
const crosshair = document.createElement('div');
crosshair.style.position = 'absolute';
crosshair.style.inset = '0';
crosshair.style.pointerEvents = 'none';
crosshair.innerHTML = `
<svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style="opacity:.25">
<line x1="50" y1="0" x2="50" y2="100" stroke="white" stroke-width="0.4"/>
<line x1="0" y1="50" x2="100" y2="50" stroke="white" stroke-width="0.4"/>
</svg>`;
el.appendChild(crosshair);


// const badge = document.createElement('div');
badge.textContent = 'Map API: not connected';
badge.style.position = 'absolute';
badge.style.right = '10px';
badge.style.bottom = '10px';
badge.style.padding = '6px 8px';
badge.style.borderRadius = '8px';
badge.style.background = 'rgba(0,0,0,.35)';
badge.style.color = 'white';
badge.style.fontSize = '12px';
badge.style.backdropFilter = 'blur(4px)';
el.appendChild(badge);
})();