(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const set = (id, value) => { el(id).textContent = value; };
  const number = value => new Intl.NumberFormat('en-US').format(value);
  const feet = value => `${value.toFixed(2)} ft`;
  const date = value => new Intl.DateTimeFormat('en-US', { month:'short', day:'numeric', year:'numeric', timeZone:'UTC' }).format(new Date(`${value}T00:00:00Z`));
  const palettes = {
    change: { min:-3, max:3, binWidth:0.5, stops:['#b3263e','#e88372','#f5f0df','#80bfd0','#2465a0'] },
    depth: { min:0, max:80, binWidth:10, stops:['#f8d77b','#8ad1c2','#3296b2','#23558e','#17294c'] }
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
  function clearDistribution(message = 'No forecast cells to summarize.') {
    el('distribution-chart').replaceChildren(); el('distribution-chart').setAttribute('hidden','');
    set('distribution-empty',message); el('distribution-empty').hidden = false;
    for (const field of ['count','area','min','median','mean','max']) set(`distribution-${field}`,'—');
  }
  function distribution() {
    const mode = el('forecast-layer').value, p = palettes[mode];
    const label = mode === 'change' ? 'Change' : 'Depth';
    set('distribution-title',`${label} distribution`);
    if (!state.collection) { clearDistribution(); return; }
    const values = state.collection.features.map(f => f.properties[`${mode}_ft`]).sort((a,b) => a-b);
    const count = values.length, midpoint = Math.floor(count/2);
    const median = count % 2 ? values[midpoint] : (values[midpoint-1]+values[midpoint])/2;
    const mean = values.reduce((sum,value) => sum+value,0)/count;
    const metricFeet = value => `${mode === 'change' && value > 0 ? '+' : ''}${feet(value)}`;
    set('distribution-count',number(count));
    set('distribution-area',`${(count*state.manifest.cell_metres**2/10000).toFixed(2)} ha`);
    for (const [field,value] of [['min',values[0]],['median',median],['mean',mean],['max',values[count-1]]]) set(`distribution-${field}`,metricFeet(value));

    const bins = Array.from({length:Math.round((p.max-p.min)/p.binWidth)},() => 0);
    // Keep the horizontal scale comparable and count outliers in the end bins.
    for (const value of values) bins[Math.max(0,Math.min(bins.length-1,Math.floor((value-p.min)/p.binWidth)))] += 1;
    const binLabel = index => index === 0 ? `Less than ${p.min+p.binWidth} ft`
      : index === bins.length-1 ? `${p.max-p.binWidth} ft or more`
      : `${p.min+index*p.binWidth} to less than ${p.min+(index+1)*p.binWidth} ft`;
    const svg = el('distribution-chart'); svg.replaceChildren();
    const add = (tag, attributes, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg',tag);
      for (const [name,value] of Object.entries(attributes)) node.setAttribute(name,String(value));
      if (text !== undefined) node.textContent = text;
      svg.append(node); return node;
    };
    add('title',{id:'distribution-chart-title'},`${label} distribution in feet across ${number(count)} cells on the map`);
    add('desc',{id:'distribution-chart-description'},`Horizontal axis: ${label.toLowerCase()} in feet. Vertical axis: cell count. ${bins.map((n,i) => `${binLabel(i)}: ${number(n)} cells`).join('; ')}. ${mode === 'change' ? 'The dashed line marks zero change.' : ''}`);
    const left = 43, right = 290, top = 24, bottom = 164;
    const highest = Math.max(...bins), magnitude = 10**Math.floor(Math.log10(highest));
    const ceiling = Math.ceil(highest/magnitude)*magnitude;
    const x = value => left+(value-p.min)/(p.max-p.min)*(right-left);
    const y = value => bottom-value/ceiling*(bottom-top);
    add('text',{x:left,y:12,class:'distribution-axis-label'},'Cell count');
    for (const tick of ceiling === 1 ? [0,1] : [0,Math.floor(ceiling/2),ceiling]) {
      add('line',{x1:left,y1:y(tick),x2:right,y2:y(tick),class:'distribution-grid'});
      add('text',{x:left-6,y:y(tick)+4,'text-anchor':'end',class:'distribution-tick'},number(tick));
    }
    bins.forEach((n,index) => {
      const start = p.min+index*p.binWidth;
      const bar = add('rect',{x:x(start)+1,y:y(n),width:(right-left)/bins.length-2,height:bottom-y(n),fill:color(start+p.binWidth/2),class:'distribution-bar'});
      const title = document.createElementNS('http://www.w3.org/2000/svg','title');
      title.textContent = `${binLabel(index)}: ${number(n)} cells`; bar.append(title);
    });
    if (mode === 'change') add('line',{x1:x(0),y1:top,x2:x(0),y2:bottom,class:'distribution-zero'});
    for (let i = 0; i <= 4; i += 1) {
      const tick = p.min+(p.max-p.min)*i/4;
      add('text',{x:x(tick),y:bottom+17,'text-anchor':'middle',class:'distribution-tick'},String(tick));
    }
    add('text',{x:(left+right)/2,y:201,'text-anchor':'middle',class:'distribution-axis-label'},`${label} (ft)`);
    el('distribution-empty').hidden = true; svg.removeAttribute('hidden');
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
    legend(); hotspots(); distribution();
  }
  async function load() {
    const frame = state.manifest.frames[Number(el('forecast-frame').value)];
    const model = state.manifest.models.find(m => m.id === el('forecast-model').value);
    const url = frame.outputs[model.id], request = ++state.request;
    state.controller?.abort(); state.controller = new AbortController();
    el('forecast-error').hidden = true; el('forecast-content').setAttribute('aria-busy','true');
    el('download-geojson').removeAttribute('href'); el('download-geojson').setAttribute('aria-disabled','true');
    clearDistribution('Loading the selected forecast…');
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
      clearDistribution();
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
    el('forecast-layer').addEventListener('change',() => { state.overlay?.setStyle(style); legend(); distribution(); });
    el('show-hotspots').addEventListener('change',hotspots);
    await load();
  }
  initialize().catch(error => { set('forecast-status','Forecast preview unavailable'); set('forecast-error',error.message); el('forecast-error').hidden = false; });
})();
