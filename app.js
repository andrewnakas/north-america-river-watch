const map = L.map('map', {
  scrollWheelZoom: false,
  worldCopyJump: true
}).setView([44, -100], 4);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

const cluster = L.markerClusterGroup();
map.addLayer(cluster);

const summaryEl = document.getElementById('summary');
const detailsEl = document.getElementById('details');
const searchEl = document.getElementById('search');
const forecastOnlyEl = document.getElementById('forecastOnly');
const sidebarEl = document.querySelector('.sidebar');
const mapEl = document.getElementById('map');

let stations = [];
let filteredStations = [];
let markers = [];

mapEl.addEventListener('mouseenter', () => map.scrollWheelZoom.enable());
mapEl.addEventListener('mouseleave', () => map.scrollWheelZoom.disable());
sidebarEl?.addEventListener('mouseenter', () => map.scrollWheelZoom.disable());

function escapeHtml(value = '') {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function haversineKm(a, b) {
  const R = 6371;
  const dLat = (b.latitude - a.latitude) * Math.PI / 180;
  const dLon = (b.longitude - a.longitude) * Math.PI / 180;
  const lat1 = a.latitude * Math.PI / 180;
  const lat2 = b.latitude * Math.PI / 180;
  const s = Math.sin(dLat / 2) ** 2 + Math.sin(dLon / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  return 2 * R * Math.asin(Math.sqrt(s));
}

function colorForStation(station) {
  if (station.country === 'CA') return '#ff8f3f';
  if (station.noaaForecast) return '#4dd0e1';
  return '#8fd14f';
}

function markerFor(station) {
  const marker = L.circleMarker([station.latitude, station.longitude], {
    radius: station.noaaForecast ? 5.5 : 4.5,
    weight: station.noaaForecast ? 2 : 1,
    color: '#0c1116',
    fillColor: colorForStation(station),
    fillOpacity: 0.9
  });
  marker.bindTooltip(`${station.name}${station.noaaForecast ? ' • official NOAA forecast' : ''}`);
  marker.on('click', () => showStation(station));
  return marker;
}

function renderMarkers() {
  cluster.clearLayers();
  markers = filteredStations.map(markerFor);
  cluster.addLayers(markers);
  summaryEl.innerHTML = `Loaded <b>${filteredStations.length.toLocaleString()}</b> river sensors. Hover the map and use your mouse wheel to zoom. <span class="legend-dot" style="background:#8fd14f"></span>USGS observed <span class="legend-dot" style="background:#4dd0e1"></span>USGS + official NOAA forecast <span class="legend-dot" style="background:#ff8f3f"></span>Canada observed`;
}

function applyFilters() {
  const q = searchEl.value.trim().toLowerCase();
  filteredStations = stations.filter((station) => {
    if (forecastOnlyEl.checked && !station.forecast) return false;
    if (!q) return true;
    const hay = [station.name, station.waterbody, station.state, station.stationId, station.network, station.country].join(' ').toLowerCase();
    return hay.includes(q);
  });
  renderMarkers();
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fetchUsgsSeries(siteNo) {
  const iv = await fetchJson(`https://waterservices.usgs.gov/nwis/iv/?format=json&sites=${encodeURIComponent(siteNo)}&parameterCd=00065,00060`);
  const dv = await fetchJson(`https://waterservices.usgs.gov/nwis/dv/?format=json&sites=${encodeURIComponent(siteNo)}&parameterCd=00065,00060&period=P30D`);
  return { iv, dv };
}

async function fetchCanadaSeries(stationNo) {
  const realtime = await fetchJson(`https://api.weather.gc.ca/collections/hydrometric-realtime/items?f=json&STATION_NUMBER=${encodeURIComponent(stationNo)}&limit=288`);
  const daily = await fetchJson(`https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?f=json&STATION_NUMBER=${encodeURIComponent(stationNo)}&limit=30`);
  return { realtime, daily };
}

function latestUsgsValue(json) {
  const series = json.value?.timeSeries || [];
  return series.map((s) => ({
    label: s.variable?.variableDescription,
    unit: s.variable?.unit?.unitCode,
    points: (s.values?.[0]?.value || []).map((v) => ({ x: v.dateTime, y: Number(v.value) })).filter((v) => Number.isFinite(v.y))
  }));
}

function latestCanadaValues(json) {
  const features = json.features || [];
  return {
    level: [...features].reverse().find((f) => f.properties.LEVEL != null)?.properties.LEVEL ?? null,
    discharge: [...features].reverse().find((f) => f.properties.DISCHARGE != null)?.properties.DISCHARGE ?? null,
    points: features.map((f) => ({ x: f.properties.DATETIME || f.properties.DATE, y: f.properties.LEVEL ?? f.properties.DISCHARGE })).filter((p) => p.y != null)
  };
}

function renderMiniChart(seriesList, title = 'River chart', subtitle = '') {
  if (!seriesList.length) return '<div class="card">No chartable series available.</div>';
  const activeSeries = seriesList.filter((s) => s?.points?.length);
  if (!activeSeries.length) return '<div class="card">No chartable series available.</div>';

  const width = 900;
  const height = 320;
  const left = 60;
  const right = 20;
  const top = 20;
  const bottom = 42;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const colors = ['#4dd0e1', '#ff8f3f', '#8fd14f', '#e66cff'];

  const pts = activeSeries.flatMap((s) => s.points || []);
  const ys = pts.map((p) => p.y).filter((v) => Number.isFinite(v));
  if (!ys.length) return '<div class="card">No chartable series available.</div>';
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yPad = Math.max((maxY - minY) * 0.08, 0.5);
  const lowY = minY - yPad;
  const highY = maxY + yPad;
  const scaleY = (y) => top + plotHeight - ((y - lowY) / ((highY - lowY) || 1)) * plotHeight;
  const scaleX = (i, len) => left + (i / Math.max(1, len - 1)) * plotWidth;

  const yTicks = Array.from({ length: 5 }, (_, i) => lowY + (i / 4) * (highY - lowY));
  const yGrid = yTicks.map((tick) => {
    const y = scaleY(tick);
    return `<line class="grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" /><text class="axis-label" x="${left - 8}" y="${y + 4}" text-anchor="end">${tick.toFixed(1)}</text>`;
  }).join('');

  const xTicks = [0, 0.33, 0.66, 1].map((frac) => ({
    x: left + frac * plotWidth,
    label: `${Math.round(frac * 100)}% horizon`
  })).map((tick) => `<text class="axis-label" x="${tick.x}" y="${height - 12}" text-anchor="middle">${tick.label}</text>`).join('');

  const paths = activeSeries.map((series, idx) => {
    const d = series.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i, series.points.length)} ${scaleY(p.y)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="2.5" />`;
  }).join('');

  const legend = activeSeries.map((series, idx) => `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${colors[idx % colors.length]}"></span>${escapeHtml(series.label)}</span>`).join('');

  return `
    <div class="chart-wrap">
      <div class="chart-title">${escapeHtml(title)}</div>
      ${subtitle ? `<div class="chart-subtitle">${escapeHtml(subtitle)}</div>` : ''}
      <svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        <line class="axis-line" x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" />
        <line class="axis-line" x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}" />
        ${yGrid}
        ${paths}
        ${xTicks}
        <text class="axis-label" x="${left}" y="${height - 28}">Time</text>
      </svg>
      <div class="chart-legend">${legend}</div>
    </div>`;
}

function formatDate(value) {
  if (!value) return 'n/a';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function summarizeForecast(noaa) {
  const forecast = noaa?.forecast;
  const observed = noaa?.observed;
  const fcPts = (forecast?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  const obPts = (observed?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  if (!forecast || !fcPts.length) {
    return {
      quality: 'Observed only',
      issued: null,
      crest: null,
      crestTime: null,
      crestFlow: null,
      primaryUnits: null,
      secondaryUnits: null,
      forecastSeries: [],
      observedSeries: []
    };
  }
  let crest = fcPts[0];
  for (const pt of fcPts) {
    if (pt.primary > crest.primary) crest = pt;
  }
  return {
    quality: 'Official NOAA forecast',
    issued: forecast.issuedTime,
    crest: crest.primary,
    crestTime: crest.validTime,
    crestFlow: crest.secondary,
    primaryUnits: forecast.primaryUnits,
    secondaryUnits: forecast.secondaryUnits,
    forecastSeries: [{ label: 'NOAA Forecast', isForecast: true, color: '#ff8f3f', dashed: true, points: fcPts.map((p) => ({ x: p.validTime, y: p.primary })) }],
    observedSeries: obPts.length ? [{ label: 'NOAA Observed', color: '#8fd14f', points: obPts.map((p) => ({ x: p.validTime, y: p.primary })) }] : []
  };
}

function renderForecastChart(seriesList, thresholds, title, subtitle) {
  const activeSeries = seriesList.filter((s) => s?.points?.length);
  if (!activeSeries.length) return '<div class="card">No data available.</div>';

  // Taller viewBox + proportional scaling = no squished text on mobile
  const W = 900, H = 480, L = 72, R = 82, T = 24, B = 70;
  const PW = W - L - R, PH = H - T - B;
  const COLORS = ['#4dd0e1', '#ff8f3f', '#8fd14f', '#e66cff'];
  const TC = { action: '#f6c90e', flood: '#ff8030', moderate: '#e06050', major: '#b060e0' };

  const allPts = activeSeries.flatMap((s) => s.points);
  const allTimes = allPts.map((p) => new Date(p.x).getTime()).filter(Number.isFinite);
  const allYs = allPts.map((p) => p.y).filter(Number.isFinite);
  const tVals = Object.values(thresholds || {}).filter((v) => v != null && Number.isFinite(+v)).map(Number);

  if (!allTimes.length || !allYs.length) return '<div class="card">No data available.</div>';

  const tMin = Math.min(...allTimes);
  const tMax = Math.max(...allTimes);
  const yMin = Math.min(...allYs, ...tVals);
  const yMax = Math.max(...allYs, ...tVals);
  const yPad = Math.max((yMax - yMin) * 0.1, 0.5);
  const lo = yMin - yPad, hi = yMax + yPad;

  const sx = (t) => L + ((t - tMin) / ((tMax - tMin) || 1)) * PW;
  const sy = (y) => T + PH - ((y - lo) / ((hi - lo) || 1)) * PH;

  // Y-axis grid — 6 ticks, brighter lines
  const yTicks = Array.from({ length: 6 }, (_, i) => lo + (i / 5) * (hi - lo));
  const yGrid = yTicks.map((v) => {
    const y = sy(v);
    return `<line stroke="#2a4560" stroke-width="1.5" x1="${L}" y1="${y}" x2="${W - R}" y2="${y}"/><text class="axis-label" x="${L - 8}" y="${y + 5}" text-anchor="end">${v.toFixed(1)}</text>`;
  }).join('');

  // X-axis date labels — 5 ticks, date only (fits mobile)
  const xTicks = Array.from({ length: 5 }, (_, i) => {
    const t = tMin + (i / 4) * (tMax - tMin);
    const d = new Date(t);
    const lbl = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    return `<text class="axis-label" x="${sx(t)}" y="${H - 18}" text-anchor="middle">${lbl}</text>`;
  }).join('');

  // Forecast shading
  const fSeries = activeSeries.find((s) => s.isForecast);
  let forecastShade = '';
  if (fSeries) {
    const fTimes = fSeries.points.map((p) => new Date(p.x).getTime()).filter(Number.isFinite);
    if (fTimes.length) {
      const fx1 = sx(Math.min(...fTimes));
      const fx2 = sx(Math.max(...fTimes));
      forecastShade = `<rect x="${fx1}" y="${T}" width="${fx2 - fx1}" height="${PH}" fill="#ff8f3f" opacity="0.07"/>`;
    }
  }

  // Flood stage threshold lines (right-side labels)
  const threshLines = Object.entries(thresholds || {})
    .filter(([, v]) => v != null && Number.isFinite(+v))
    .map(([key, val]) => {
      const y = sy(+val);
      const c = TC[key] || '#888';
      const lbl = key.charAt(0).toUpperCase() + key.slice(1);
      return `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="${c}" stroke-width="2" stroke-dasharray="6,3"/>
              <text x="${W - R + 5}" y="${y + 5}" font-size="13" fill="${c}" font-weight="600">${lbl}</text>`;
    }).join('');

  // Current stage horizontal reference line + dot
  const lastObsPt = activeSeries.filter((s) => !s.isForecast).find((s) => s.points.length)?.points.at(-1);
  const currentVal = lastObsPt ? lastObsPt.y : null;
  const currentValTime = lastObsPt ? new Date(lastObsPt.x).getTime() : null;
  const currentValLine = (currentVal != null && Number.isFinite(currentVal)) ? `
    <line x1="${L}" y1="${sy(currentVal)}" x2="${W - R}" y2="${sy(currentVal)}" stroke="#7ec8ff" stroke-width="2" stroke-dasharray="3,4"/>
    <rect x="0" y="${sy(currentVal) - 10}" width="${L - 4}" height="16" rx="3" fill="#0f1b2a"/>
    <text x="${L - 6}" y="${sy(currentVal) + 5}" text-anchor="end" font-size="13" fill="#7ec8ff" font-weight="700">${currentVal.toFixed(1)}</text>` : '';
  const currentDot = (currentVal != null && currentValTime != null) ? `
    <circle cx="${sx(currentValTime)}" cy="${sy(currentVal)}" r="6" fill="#4dd0e1" stroke="#e5eef7" stroke-width="2"/>` : '';

  // "Now" vertical line — bright white, prominent
  const now = Date.now();
  const nowLine = (now >= tMin && now <= tMax) ? `
    <line x1="${sx(now)}" y1="${T}" x2="${sx(now)}" y2="${T + PH}" stroke="#e5eef7" stroke-width="2" stroke-dasharray="5,4" opacity="0.7"/>
    <text x="${sx(now) + 5}" y="${T + 16}" font-size="13" fill="#e5eef7" font-weight="600" opacity="0.85">Now</text>` : '';

  // Data series paths
  const paths = activeSeries.map((series, idx) => {
    const c = series.color || COLORS[idx % COLORS.length];
    const da = series.dashed ? 'stroke-dasharray="8,4"' : '';
    const d = series.points
      .map((p) => [new Date(p.x).getTime(), p.y])
      .filter(([t, y]) => Number.isFinite(t) && Number.isFinite(y))
      .map(([t, y], i) => `${i === 0 ? 'M' : 'L'} ${sx(t)} ${sy(y)}`)
      .join(' ');
    return d ? `<path d="${d}" fill="none" stroke="${c}" stroke-width="3" ${da}/>` : '';
  }).join('');

  const seriesLegend = activeSeries.map((s, idx) => {
    const c = s.color || COLORS[idx % COLORS.length];
    return `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${c}"></span>${escapeHtml(s.label)}</span>`;
  }).join('');
  const threshLegend = Object.entries(thresholds || {})
    .filter(([, v]) => v != null && Number.isFinite(+v))
    .map(([key, val]) => {
      const c = TC[key] || '#888';
      return `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${c};border-radius:2px"></span>${escapeHtml(key.charAt(0).toUpperCase() + key.slice(1))}: ${(+val).toFixed(1)} ft</span>`;
    }).join('');

  return `
    <div class="chart-wrap">
      <div class="chart-title">${escapeHtml(title)}</div>
      ${subtitle ? `<div class="chart-subtitle">${escapeHtml(subtitle)}</div>` : ''}
      <svg class="chart chart-forecast" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        <line class="axis-line" x1="${L}" y1="${T}" x2="${L}" y2="${T + PH}"/>
        <line class="axis-line" x1="${L}" y1="${T + PH}" x2="${W - R}" y2="${T + PH}"/>
        ${yGrid}
        ${forecastShade}
        ${threshLines}
        ${currentValLine}
        ${paths}
        ${currentDot}
        ${nowLine}
        ${xTicks}
      </svg>
      <div class="chart-legend">${seriesLegend}${threshLegend}</div>
    </div>`;
}

async function showStation(station) {
  detailsEl.innerHTML = `<div class="card">Loading ${escapeHtml(station.name)}…</div>`;
  try {
    let body = '';
    if (station.country === 'US') {
      const [usgs, noaa] = await Promise.all([
        fetchUsgsSeries(station.stationId),
        station.noaaLid ? fetchJson(`data/noaa-forecast/${station.noaaLid}.json`).catch(() => null) : Promise.resolve(null)
      ]);
      const liveSeries = latestUsgsValue(usgs.iv);
      const dailySeries = latestUsgsValue(usgs.dv);
      const currentStageSeries = liveSeries.find((s) => /gage height/i.test(s.label));
      const currentFlowSeries = liveSeries.find((s) => /discharge/i.test(s.label));
      const currentStage = currentStageSeries?.points.at(-1)?.y;
      const currentFlow = currentFlowSeries?.points.at(-1)?.y;
      const forecast = summarizeForecast(noaa);
      const thresholds = {
        action: station.noaaAction ?? null,
        flood: station.noaaFlood ?? null,
        moderate: station.noaaModerate ?? null,
        major: station.noaaMajor ?? null
      };
      const usgsStageSeries = currentStageSeries ? { label: 'USGS Stage', color: '#4dd0e1', points: currentStageSeries.points } : null;
      const chartSeries = [usgsStageSeries, ...forecast.observedSeries, ...forecast.forecastSeries].filter(Boolean);
      const hydrograph = forecast.forecastSeries.length
        ? renderForecastChart(chartSeries, thresholds, '15-Day Hydrograph — Observed + NOAA Forecast', `Issued: ${formatDate(forecast.issued)} • ${escapeHtml(forecast.quality)}`)
        : renderMiniChart([usgsStageSeries, currentFlowSeries, ...dailySeries.slice(0, 1)].filter(Boolean), 'Observed hydrograph', 'USGS observed and historical context.');
      body = `
        <h2>${escapeHtml(station.name)}</h2>
        <div class="meta">USGS observed • ${station.state} • ${escapeHtml(station.stationId)}${station.noaaLid ? ` • NOAA ${escapeHtml(station.noaaLid)}` : ''}</div>
        <div class="kv">
          <div class="card"><b>Current stage</b><br>${currentStage ?? 'n/a'} ${escapeHtml(currentStageSeries?.unit || '')}</div>
          <div class="card"><b>Current discharge</b><br>${currentFlow ?? 'n/a'} ${escapeHtml(currentFlowSeries?.unit || '')}</div>
          <div class="card"><b>Best forecast</b><br>${escapeHtml(forecast.quality)}</div>
          <div class="card"><b>Forecast issued</b><br>${escapeHtml(formatDate(forecast.issued))}</div>
          <div class="card"><b>Forecast crest</b><br>${forecast.crest ?? 'n/a'} ${escapeHtml(forecast.primaryUnits || '')}</div>
          <div class="card"><b>Crest time</b><br>${escapeHtml(formatDate(forecast.crestTime))}</div>
        </div>
        <p><a href="https://waterdata.usgs.gov/monitoring-location/${encodeURIComponent(station.stationId)}/" target="_blank" rel="noreferrer">USGS station page</a>${station.noaaHydrographUrl ? ` • <a href="${station.noaaHydrographUrl}" target="_blank" rel="noreferrer">NOAA hydrograph</a>` : ''}</p>
        ${hydrograph}
        <p>Forecast priority here is: <b>official NOAA forecast</b> when matched, then USGS observed/historical context. Click anywhere on the map to snap to the nearest river sensor.</p>`;
    } else {
      const series = await fetchCanadaSeries(station.stationId);
      const latest = latestCanadaValues(series.realtime);
      const dailyPoints = (series.daily.features || []).map((f) => ({ x: f.properties.DATE, y: f.properties.LEVEL ?? f.properties.DISCHARGE })).filter((p) => p.y != null);
      body = `
        <h2>${escapeHtml(station.name)}</h2>
        <div class="meta">Environment Canada observed • ${station.state} • ${escapeHtml(station.stationId)}</div>
        <div class="kv">
          <div class="card"><b>Current level</b><br>${latest.level ?? 'n/a'}</div>
          <div class="card"><b>Current discharge</b><br>${latest.discharge ?? 'n/a'}</div>
          <div class="card"><b>Historical daily mean</b><br>last 30 values</div>
          <div class="card"><b>Forecast quality</b><br>Observed only nationally</div>
        </div>
        <p><a href="https://wateroffice.ec.gc.ca/report/real_time_e.html?stn=${encodeURIComponent(station.stationId)}" target="_blank" rel="noreferrer">Environment Canada station page</a></p>
        ${renderMiniChart([{ label: 'Realtime', points: latest.points }, { label: 'Daily mean', points: dailyPoints }], 'Observed hydrograph', 'Environment Canada realtime plus recent daily mean history.')}`;
    }
    detailsEl.innerHTML = body;
  } catch (error) {
    detailsEl.innerHTML = `<div class="card">Could not load sensor details. ${escapeHtml(String(error.message || error))}</div>`;
  }
}

function findNearest(latlng) {
  const point = { latitude: latlng.lat, longitude: latlng.lng };
  let nearest = null;
  let best = Infinity;
  for (const station of filteredStations.length ? filteredStations : stations) {
    const dist = haversineKm(point, station);
    if (dist < best) {
      best = dist;
      nearest = station;
    }
  }
  return nearest;
}

map.on('click', (event) => {
  const nearest = findNearest(event.latlng);
  if (nearest) {
    showStation(nearest);
    map.flyTo([nearest.latitude, nearest.longitude], Math.max(map.getZoom(), 8), { duration: 0.8 });
  }
});

searchEl.addEventListener('input', applyFilters);
forecastOnlyEl.addEventListener('change', applyFilters);

const [stationData, sourceData] = await Promise.all([
  fetchJson('data/stations.json'),
  fetchJson('data/sources.json')
]);
stations = stationData;
filteredStations = stationData;
renderMarkers();
summaryEl.innerHTML += `<div style="margin-top:8px;color:#9fb4c7">Built ${new Date(sourceData.generatedAt).toLocaleString()} • ${sourceData.usgsStations.toLocaleString()} USGS • ${sourceData.canadaStations.toLocaleString()} Canada • ${sourceData.matchedForecastStations.toLocaleString()} official NOAA forecast matches</div>`;
