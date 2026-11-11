(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const set = (id, value) => { el(id).textContent = value; };
  const number = value => new Intl.NumberFormat('en-US').format(value);
  const feet = value => `${value.toFixed(2)} ft`;
  const date = value => new Intl.DateTimeFormat('en-US', { month:'short', day:'numeric', year:'numeric', timeZone:'UTC' }).format(new Date(`${value}T00:00:00Z`));
  const palettes = {
    change: { min:-3, max:3, stops:['#b3263e','#e88372','#f5f0df','#80bfd0','#2465a0'] },
    depth: { min:0, max:80, stops:['#f8d77b','#8ad1c2','#3296b2','#23558e','#17294c'] }
  };
  const state = { manifest:null, map:null, overlay:null, hotspots:null, collection:null, fitted:false, request:0, controller:null, cache:new Map() };
  function color(value) {
    const p = palettes[el('forecast-layer').value];
    const position = Math.max(0, Math.min(1, (value-p.min)/(p.max-p.min))) * (p.stops.length-1);
    const index = Math.min(p.stops.length-2, Math.floor(position));
    const fraction = position-index;
    const rgb = p.stops.slice(index,index+2).map(hex => [1,3,5].map(i => parseInt(hex.slice(i,i+2),16)));
    return `rgb(${rgb[0].map((v,i) => Math.round(v+(rgb[1][i]-v)*fraction)).join(',')})`;
  }
  function style(feature) {
    return { stroke:false, fillColor:color(feature.properties[el('forecast-layer').value === 'depth' ? 'depth_ft' : 'change_ft']), fillOpacity:0.86 };
  }
  function legend() {
    const mode = el('forecast-layer').value, p = palettes[mode];
    el('legend-gradient').style.background = `linear-gradient(to right,${p.stops.join(',')})`;
    set('legend-min', `${p.min} ft or less`); set('legend-max', `${p.max} ft or more`);
    set('legend-description', mode === 'change' ? 'Predicted change · red: shallower · blue: deeper. Colors use a fixed scale across all dates and models.' : 'Predicted depth below low tide · yellow: shallow · dark blue: deep. Colors use a fixed scale across all dates and models.');
  }
  function popup(feature) {
    const div = document.createElement('div'); div.className = 'forecast-popup';
    const title = document.createElement('h3'); title.textContent = 'Predicted 10 m cell'; div.append(title);
    const rows = [ ['Predicted depth', feet(feature.properties.depth_ft)], ['Predicted change', feet(feature.properties.change_ft)], ['From survey', date(state.collection.properties.from_survey)], ['Predicted for', date(state.collection.properties.predicted_for)] ];
    for (const [label,value] of rows) { const p = document.createElement('p'); p.textContent = `${label}: ${value}`; div.append(p); }
    return div;
  }
  function hotspots() {
    state.hotspots.clearLayers();
    if (!el('show-hotspots').checked || !state.collection) return;
    const groups = new Map();
    for (const feature of state.collection.features) {
      if (feature.properties.change_ft > -1) continue;
      const ring = feature.geometry.coordinates[0].slice(0,4);
      const lon = ring.reduce((sum,p) => sum+p[0],0)/4, lat = ring.reduce((sum,p) => sum+p[1],0)/4;
      const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
      const group = groups.get(key) || {lat:0,lon:0,n:0}; group.lat += lat; group.lon += lon; group.n += 1; groups.set(key,group);
    }
    for (const group of groups.values()) {
      L.circleMarker([group.lat/group.n,group.lon/group.n], {radius:5,color:'#8d1332',fillColor:'#ffbfbc',fillOpacity:0.95,weight:2})
        .bindPopup(`${number(group.n)} nearby cells predicted to lose at least 1 ft.`).addTo(state.hotspots);
    }
  }
  function validate(collection, frame) {
    if (collection.type !== 'FeatureCollection' || !Array.isArray(collection.features) || !collection.features.length || collection.properties?.from_survey !== frame.from_survey || collection.properties?.predicted_for !== frame.predicted_for) throw new Error('The forecast file does not match the selected dates.');
    for (const f of collection.features) {
      const ring = f.geometry?.coordinates?.[0];
      if (f.type !== 'Feature' || f.geometry?.type !== 'Polygon' || !Array.isArray(ring) || ring.length !== 5 || !Number.isFinite(f.properties?.depth_ft) || !Number.isFinite(f.properties?.change_ft) || !ring.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)) || ring[0][0] !== ring[4][0] || ring[0][1] !== ring[4][1]) throw new Error('The forecast file contains an invalid prediction cell.');
    }
  }
  function show(collection, model, frame, url) {
    if (state.overlay) state.overlay.remove();
    state.collection = collection;
    state.overlay = L.geoJSON(collection, {style, onEachFeature:(feature,layer) => layer.bindPopup(() => popup(feature))}).addTo(state.map);
    if (!state.fitted) { state.map.fitBounds(state.overlay.getBounds(), {padding:[24,24],maxZoom:16}); state.fitted = true; }
    const rows = collection.features, count = rows.length;
    const depth = rows.reduce((sum,f) => sum+f.properties.depth_ft,0)/count;
    const change = rows.reduce((sum,f) => sum+f.properties.change_ft,0)/count;
    const hotCount = rows.filter(f => f.properties.change_ft <= -1).length;
    const squareMetres = hotCount * state.manifest.cell_metres ** 2;
    set('cell-count',number(count)); set('mean-depth',feet(depth)); set('mean-change',`${change > 0 ? '+' : ''}${feet(change)}`);
    set('hotspot-area',`${(squareMetres/10000).toFixed(2)} ha`);
    el('hotspot-area').title = `${number(hotCount)} cells · ${number(squareMetres)} m² predicted to lose at least 1 ft`;
    set('model-name',model.label); set('model-score',Number.isFinite(model.mae_ft) ? model.mae_ft.toFixed(3) : 'Unavailable');
    set('forecast-status',`Historical forecast preview · ${model.label} · ${number(count)} cells`);
    const download = el('download-geojson'); download.href = url; download.download = `anchorage-${model.id}-${frame.from_survey}.geojson`; download.setAttribute('aria-disabled','false');
    legend(); hotspots();
  }
  async function load() {
    const frame = state.manifest.frames[Number(el('forecast-frame').value)];
    const model = state.manifest.models.find(m => m.id === el('forecast-model').value);
    const url = frame.outputs[model.id], request = ++state.request;
    state.controller?.abort(); state.controller = new AbortController();
    el('forecast-error').hidden = true; el('forecast-content').setAttribute('aria-busy','true');
    el('download-geojson').removeAttribute('href'); el('download-geojson').setAttribute('aria-disabled','true');
    set('frame-origin',date(frame.from_survey)); set('frame-target',`Predicting ${date(frame.predicted_for)} · 28-day forecast`);
    set('forecast-status',`Loading ${model.label}…`);
    try {
      if (!url) throw new Error('This model has no output for the selected date.');
      let collection = state.cache.get(url);
      if (!collection) {
        const response = await fetch(url,{signal:state.controller.signal});
        if (!response.ok) throw new Error(`Forecast could not be loaded (${response.status}).`);
        collection = await response.json(); validate(collection,frame); state.cache.set(url,collection);
      }
      if (request !== state.request) return;
      show(collection,model,frame,url);
    } catch (error) {
      if (error.name === 'AbortError' || request !== state.request) return;
      state.overlay?.remove(); state.overlay = null; state.collection = null; state.hotspots.clearLayers();
      for (const id of ['cell-count','mean-depth','mean-change','hotspot-area','model-name','model-score']) set(id,'—');
      set('forecast-status','Forecast unavailable'); set('forecast-error',`${error.message} Please choose another date or reload the page.`); el('forecast-error').hidden = false;
    } finally { if (request === state.request) el('forecast-content').setAttribute('aria-busy','false'); }
  }
  async function initialize() {
    const response = await fetch('/static/data/anchorage/manifest.json');
    if (!response.ok) throw new Error(`Forecast index could not be loaded (${response.status}).`);
    const manifest = await response.json();
    if (manifest.schema_version !== 1 || !manifest.models?.length || !manifest.frames?.length || !(manifest.cell_metres > 0)) throw new Error('The forecast index is invalid.');
    state.manifest = manifest;
    for (const model of manifest.models) { const option = document.createElement('option'); option.value = model.id; option.textContent = model.label; el('forecast-model').append(option); }
    el('forecast-frame').max = manifest.frames.length-1; el('forecast-frame').value = manifest.frames.length-1;
    el('forecast-content').hidden = false;
    state.map = L.map('map',{preferCanvas:true}).setView([61.24,-149.9],14);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'})
      .on('tileerror',() => { el('basemap-status').hidden = false; }).addTo(state.map);
    L.control.scale({imperial:false}).addTo(state.map); state.hotspots = L.layerGroup().addTo(state.map);
    el('forecast-model').addEventListener('change',load); el('forecast-frame').addEventListener('input',load);
    el('forecast-layer').addEventListener('change',() => { state.overlay?.setStyle(style); legend(); });
    el('show-hotspots').addEventListener('change',hotspots);
    await load();
  }
  initialize().catch(error => { set('forecast-status','Forecast preview unavailable'); set('forecast-error',error.message); el('forecast-error').hidden = false; });
})();
