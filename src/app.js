const DATA_BASES = [
  'data',
  'https://cdn.jsdelivr.net/gh/andrewnakas/north-america-river-watch@sensor-data/data'
];

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
const startupStatus = { phase: 'boot' };
const detailsEl = document.getElementById('details');
const searchEl = document.getElementById('search');
const forecastOnlyEl = document.getElementById('forecastOnly');
const mlForecastOnlyEl = document.getElementById('mlForecastOnly');
const sidebarEl = document.querySelector('.sidebar');
const mapEl = document.getElementById('map');

let stations = [];
let filteredStations = [];
let markers = [];
let detailMap = null;
let activeStationId = null;
let mlForecastSummary = null;
let mlForecastByStationId = new Map();

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
  if (station.mlForecast) return '#ff6b9f';
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
  marker.bindTooltip(`${station.name}${station.mlForecast ? ' • Montana ML runoff forecast' : station.noaaForecast ? ' • official NOAA forecast' : ''}`);
  marker.on('click', () => showStation(station));
  return marker;
}

function renderMarkers() {
  cluster.clearLayers();
  markers = filteredStations.map(markerFor);
  cluster.addLayers(markers);
  const mlCount = filteredStations.filter((station) => station.mlForecast).length;
  summaryEl.innerHTML = `Loaded <b>${filteredStations.length.toLocaleString()}</b> river sensors. Hover the map and use your mouse wheel to zoom. <span class="legend-dot" style="background:#8fd14f"></span>USGS observed <span class="legend-dot" style="background:#4dd0e1"></span>USGS + official NOAA forecast <span class="legend-dot" style="background:#ff6b9f"></span>Montana ML runoff <span class="legend-dot" style="background:#ff8f3f"></span>Canada observed${mlCount ? ` <span class="summary-ml-line">• ${mlCount} ML forecast stations highlighted</span>` : ''}`;
}

function stationHasForecastCapability(station) {
  return Boolean(station?.forecast || station?.noaaForecast || station?.noaaForecastPath || station?.mlForecast);
}

function stationMatchesMlForecast(station) {
  return Boolean(station?.mlForecast || mlForecastByStationId.has(String(station?.stationId || station?.id || '')));
}

function renderMlForecastCard(station) {
  const forecast = mlForecastByStationId.get(String(station.stationId || station.id || ''));
  if (!forecast) return '';
  const preds = forecast.predictions || [];
  const first = preds[0] || null;
  const peak = preds.reduce((best, point) => (!best || point.predicted_discharge_cfs > best.predicted_discharge_cfs ? point : best), null);
  const latest = forecast.latest_observed_discharge_cfs;
  const delta = first && Number.isFinite(latest) ? first.predicted_discharge_cfs - latest : null;
  const trend = delta == null ? 'n/a' : delta > 5 ? 'rising' : delta < -5 ? 'falling' : 'steady';
  const chartSeries = [
    latest != null ? { label: 'Latest observed discharge', style: 'observed', points: [{ x: new Date(forecast.generated_at).toISOString(), y: latest }] } : null,
    preds.length ? { label: 'Montana ML forecast discharge', style: 'forecast', points: preds.map((p) => ({ x: p.date, y: p.predicted_discharge_cfs })) } : null
  ].filter(Boolean);
  const dayCards = preds.slice(0, 7).map((p) => `
    <div class="card">
      <div class="date">${escapeHtml(formatAxisDate(p.date))}</div>
      <div class="value">${Number(p.predicted_discharge_cfs).toFixed(1)} cfs</div>
      <div class="meta">${p.temp_c_mean ?? 'n/a'}°C mean • ${p.precip_mm ?? 'n/a'} mm precip</div>
      <div class="meta">degree day ${p.degree_day_c ?? 'n/a'} • SWE ${p.snotel_wteq_in ?? 'n/a'} in</div>
    </div>`).join('');

  return `
    <div class="forecast-ml-summary">
      <div class="header-line"><span class="ml-badge">ML</span><h3>Montana runoff forecast</h3></div>
      <p class="detail-intro">Experimental 7-day discharge forecast for this Yellowstone/Gallatin corridor gauge. This is a modeled runoff layer, separate from official NOAA hydrographs.</p>
      <div class="forecast-summary-grid">
        <div class="card"><b>Latest observed</b><br>${latest ?? 'n/a'} cfs</div>
        <div class="card"><b>Next predicted</b><br>${first ? first.predicted_discharge_cfs.toFixed(1) : 'n/a'} cfs</div>
        <div class="card"><b>Peak predicted</b><br>${peak ? peak.predicted_discharge_cfs.toFixed(1) : 'n/a'} cfs</div>
        <div class="card"><b>Peak date</b><br>${peak ? escapeHtml(formatAxisDate(peak.date)) : 'n/a'}</div>
        <div class="card"><b>Near-term signal</b><br>${trend}</div>
        <div class="card"><b>Model scope</b><br>${escapeHtml(forecast.target_group || 'montana')}</div>
      </div>
      ${chartSeries.length ? renderMiniChart(chartSeries, 'Montana ML discharge forecast', 'Latest observed discharge plus 7-day modeled discharge outlook.', { compact: true }) : ''}
      <div class="forecast-day-grid">${dayCards}</div>
      <p class="note-outro">Generated ${escapeHtml(formatDate(forecast.generated_at))}. Source weather: Open-Meteo forecast. Use this as an experimental planning layer, not a safety-critical forecast.</p>
    </div>`;
}

function applyFilters() {
  const q = searchEl.value.trim().toLowerCase();
  filteredStations = stations.filter((station) => {
    if (forecastOnlyEl.checked && !stationHasForecastCapability(station)) return false;
    if (mlForecastOnlyEl?.checked && !stationMatchesMlForecast(station)) return false;
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

async function fetchJsonWithFallback(relativePath) {
  let lastError = null;
  const attempts = [];
  for (const base of DATA_BASES) {
    const url = `${base.replace(/\/$/, '')}/${relativePath.replace(/^\//, '')}`;
    try {
      const data = await fetchJson(url);
      startupStatus.lastSuccess = url;
      return data;
    } catch (error) {
      attempts.push(`${url} → ${String(error?.message || error)}`);
      lastError = error;
    }
  }
  const err = new Error(`Unable to load ${relativePath}. Attempts: ${attempts.join(' | ')}`);
  err.attempts = attempts;
  throw err;
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

function formatAxisDate(value, includeHour = false) {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return 'n/a';
  return includeHour
    ? d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric' })
    : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function estimateSeriesStepMs(points = []) {
  if (!points || points.length < 2) return 3600000;
  const deltas = [];
  for (let i = 1; i < points.length; i += 1) {
    const prev = new Date(points[i - 1].x).getTime();
    const curr = new Date(points[i].x).getTime();
    if (Number.isFinite(prev) && Number.isFinite(curr) && curr > prev) deltas.push(curr - prev);
  }
  if (!deltas.length) return 3600000;
  deltas.sort((a, b) => a - b);
  return deltas[Math.floor(deltas.length / 2)] || 3600000;
}

function buildSyntheticForecastSeries(observedSeries, label = 'Synthetic hydrograph forecast') {
  const points = observedSeries?.points || [];
  if (points.length < 6) return null;

  const recent = points.slice(-12);
  const recentPairs = [];
  for (let i = 1; i < recent.length; i += 1) {
    const prev = recent[i - 1];
    const curr = recent[i];
    const prevMs = new Date(prev.x).getTime();
    const currMs = new Date(curr.x).getTime();
    if (!Number.isFinite(prevMs) || !Number.isFinite(currMs) || currMs <= prevMs) continue;
    recentPairs.push({
      hours: (currMs - prevMs) / 3600000,
      delta: curr.y - prev.y
    });
  }
  if (!recentPairs.length) return null;

  const hoursTotal = recentPairs.reduce((sum, item) => sum + item.hours, 0) || 1;
  const weightedSlopePerHour = recentPairs.reduce((sum, item) => sum + item.delta, 0) / hoursTotal;
  const stepMs = Math.max(estimateSeriesStepMs(points), 3600000);
  const lastPoint = points.at(-1);
  const lastTime = new Date(lastPoint.x).getTime();
  if (!Number.isFinite(lastTime)) return null;

  const synthetic = [];
  let currentValue = lastPoint.y;
  for (let step = 1; step <= 12; step += 1) {
    const progress = step / 12;
    const damp = Math.max(0.15, 1 - progress * 0.7);
    currentValue += weightedSlopePerHour * (stepMs / 3600000) * damp;
    synthetic.push({
      x: new Date(lastTime + stepMs * step).toISOString(),
      y: Number(currentValue.toFixed(2))
    });
  }

  return synthetic.length ? { label, style: 'forecast', synthetic: true, points: synthetic } : null;
}

function renderMiniChart(seriesList, title = 'River chart', subtitle = '', options = {}) {
  if (!seriesList.length) return '<div class="card">No chartable series available.</div>';

  const palette = {
    observed: '#4dd0e1',
    forecast: '#ffb14a',
    history: '#8fd14f',
    secondary: '#e66cff',
    now: '#ffd84d'
  };

  const activeSeries = seriesList.map((series, idx) => ({
    ...series,
    style: series.style || (idx === 0 ? 'observed' : idx === 1 ? 'forecast' : idx === 2 ? 'history' : 'secondary'),
    points: (series?.points || []).map((point, index) => ({
      ...point,
      xMs: Number.isFinite(new Date(point.x).getTime()) ? new Date(point.x).getTime() : index
    })).filter((point) => Number.isFinite(point.y))
  })).filter((s) => s.points.length);

  if (!activeSeries.length) return '<div class="card">No chartable series available.</div>';

  const nowMs = Date.now();
  const width = 900;
  const height = options.compact ? 300 : 360;
  const left = 54;
  const right = 18;
  const top = 18;
  const bottom = 52;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const pts = activeSeries.flatMap((s) => s.points || []);
  const ys = pts.map((p) => p.y).filter((v) => Number.isFinite(v));
  const xs = pts.map((p) => p.xMs).filter((v) => Number.isFinite(v));
  if (!ys.length || !xs.length) return '<div class="card">No chartable series available.</div>';

  const rawMinX = Math.min(...xs);
  const rawMaxX = Math.max(...xs);
  const rangeX = Math.max(rawMaxX - rawMinX, 1);
  const minX = Math.min(rawMinX, nowMs);
  const maxX = Math.max(rawMaxX, nowMs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yPad = Math.max((maxY - minY) * 0.1, 0.5);
  const lowY = minY - yPad;
  const highY = maxY + yPad;

  const scaleY = (y) => top + plotHeight - ((y - lowY) / ((highY - lowY) || 1)) * plotHeight;
  const scaleX = (x) => left + ((x - minX) / ((maxX - minX) || 1)) * plotWidth;

  const yTicks = Array.from({ length: 5 }, (_, i) => lowY + (i / 4) * (highY - lowY));
  const yGrid = yTicks.map((tick) => {
    const y = scaleY(tick);
    return `<line class="grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" /><text class="axis-label axis-label-y" x="${left - 10}" y="${y + 5}" text-anchor="end">${tick.toFixed(1)}</text>`;
  }).join('');

  const xTickCount = options.compact ? 3 : 4;
  const xTicks = Array.from({ length: xTickCount }, (_, i) => {
    const frac = xTickCount === 1 ? 0 : i / (xTickCount - 1);
    const xValue = minX + frac * (maxX - minX);
    return {
      x: scaleX(xValue),
      label: formatAxisDate(xValue, rangeX < 5 * 86400000)
    };
  }).map((tick) => `<text class="axis-label axis-label-x" x="${tick.x}" y="${height - 10}" text-anchor="middle">${escapeHtml(tick.label)}</text>`).join('');

  const areaPaths = activeSeries.map((series, idx) => {
    if (!['observed', 'forecast'].includes(series.style)) return '';
    const first = series.points[0];
    const last = series.points.at(-1);
    if (!first || !last) return '';
    const line = series.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.xMs)} ${scaleY(p.y)}`).join(' ');
    const area = `${line} L ${scaleX(last.xMs)} ${top + plotHeight} L ${scaleX(first.xMs)} ${top + plotHeight} Z`;
    return `<path d="${area}" fill="url(#chartArea${idx})" class="chart-area" />`;
  }).join('');

  const paths = activeSeries.map((series) => {
    const d = series.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.xMs)} ${scaleY(p.y)}`).join(' ');
    const stroke = palette[series.style] || palette.secondary;
    const dash = series.style === 'forecast' ? '7 5' : '';
    return `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ''} />`;
  }).join('');

  const peakPoint = activeSeries.flatMap((series) => (series.style === 'forecast' ? series.points.map((point) => ({ ...point, series })) : [])).reduce((best, point) => {
    if (!best || point.y > best.y) return point;
    return best;
  }, null);

  const peakLabel = peakPoint
    ? `<g class="peak-callout"><circle cx="${scaleX(peakPoint.xMs)}" cy="${scaleY(peakPoint.y)}" r="5.5" fill="${palette.forecast}" stroke="#08111b" stroke-width="2" /><rect class="label-pill" x="${Math.min(scaleX(peakPoint.xMs) + 4, width - 114)}" y="${Math.max(scaleY(peakPoint.y) - 24, top + 2)}" width="108" height="20" rx="6" ry="6" /><text class="peak-label" x="${Math.min(scaleX(peakPoint.xMs) + 12, width - 106)}" y="${Math.max(scaleY(peakPoint.y) - 10, top + 16)}">Peak ${peakPoint.y.toFixed(1)}</text></g>`
    : '';

  const forecastStartMs = activeSeries.filter((series) => series.style === 'forecast').map((series) => series.points[0]?.xMs).filter(Number.isFinite).sort((a, b) => a - b)[0];
  const forecastDivider = Number.isFinite(forecastStartMs)
    ? `<line class="forecast-divider" x1="${scaleX(forecastStartMs)}" y1="${top}" x2="${scaleX(forecastStartMs)}" y2="${top + plotHeight}" /><rect class="label-pill" x="${Math.min(scaleX(forecastStartMs) + 4, width - 92)}" y="${top + 10}" width="78" height="20" rx="6" ry="6" /><text class="forecast-divider-label" x="${Math.min(scaleX(forecastStartMs) + 12, width - 84)}" y="${top + 24}">Forecast</text>`
    : '';

  const endPoints = activeSeries.map((series) => {
    const point = series.points.at(-1);
    if (!point) return '';
    const stroke = palette[series.style] || palette.secondary;
    return `<circle cx="${scaleX(point.xMs)}" cy="${scaleY(point.y)}" r="4.5" fill="${stroke}" stroke="#08111b" stroke-width="2" />`;
  }).join('');

  const nowInsideRange = nowMs >= minX && nowMs <= maxX;
  const nowLine = nowInsideRange ? `<line class="now-line" x1="${scaleX(nowMs)}" y1="${top}" x2="${scaleX(nowMs)}" y2="${top + plotHeight}" /><rect class="label-pill" x="${Math.min(scaleX(nowMs) + 4, width - 56)}" y="${top + 2}" width="42" height="18" rx="6" ry="6" /><text class="now-label" x="${Math.min(scaleX(nowMs) + 12, width - 48)}" y="${top + 15}">Now</text>` : '';

  const legend = activeSeries.map((series) => {
    const swatch = palette[series.style] || palette.secondary;
    const klass = series.style === 'forecast' ? 'chart-legend-line chart-legend-line-forecast' : 'chart-legend-line';
    return `<span class="chart-legend-item"><span class="${klass}" style="--legend-color:${swatch}"></span>${escapeHtml(series.label)}</span>`;
  }).join('');

  const nowLegend = nowInsideRange ? `<span class="chart-legend-item"><span class="chart-legend-line chart-legend-line-now" style="--legend-color:${palette.now}"></span>Now</span>` : '';

  return `
    <div class="chart-wrap ${options.compact ? 'chart-wrap-compact' : ''}">
      <div class="chart-head">
        <div>
          <div class="chart-title">${escapeHtml(title)}</div>
          ${subtitle ? `<div class="chart-subtitle">${escapeHtml(subtitle)}</div>` : ''}
        </div>
      </div>
      <svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(title)}">
        <defs>
          ${activeSeries.map((series, idx) => {
            const stroke = palette[series.style] || palette.secondary;
            return `<linearGradient id="chartArea${idx}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${stroke}" stop-opacity="0.22" /><stop offset="100%" stop-color="${stroke}" stop-opacity="0.02" /></linearGradient>`;
          }).join('')}
        </defs>
        <line class="axis-line" x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" />
        <line class="axis-line" x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}" />
        ${yGrid}
        ${forecastDivider}
        ${nowLine}
        ${areaPaths}
        ${paths}
        ${endPoints}
        ${peakLabel}
        ${xTicks}
        <text class="axis-label axis-label-bottom" x="${left}" y="${height - 30}">Time</text>
      </svg>
      <div class="chart-legend">${legend}${nowLegend}</div>
    </div>`;
}

function formatDate(value) {
  if (!value) return 'n/a';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function renderSiteMapCard(station) {
  return `
    <div class="site-map-card card">
      <div class="site-map-head">
        <div>
          <b>Site map</b><br>
          <span class="site-map-subtitle">Station location and nearby context</span>
        </div>
        <div class="site-map-coords">${station.latitude.toFixed(4)}, ${station.longitude.toFixed(4)}</div>
      </div>
      <div id="detailSiteMap" class="detail-site-map" aria-label="Station site map"></div>
    </div>`;
}

function mountDetailMap(station) {
  const target = document.getElementById('detailSiteMap');
  if (!target) return;
  if (detailMap) {
    detailMap.remove();
    detailMap = null;
  }
  detailMap = L.map(target, {
    zoomControl: false,
    attributionControl: false,
    dragging: true,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    tap: false
  }).setView([station.latitude, station.longitude], 11);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(detailMap);

  L.circleMarker([station.latitude, station.longitude], {
    radius: 8,
    weight: 2,
    color: '#08111b',
    fillColor: colorForStation(station),
    fillOpacity: 0.95
  }).addTo(detailMap);
}

function summarizeForecast(noaa) {
  const stageflow = noaa?.stageflow || noaa || {};
  const observed = stageflow?.observed;
  const forecast = stageflow?.forecast;
  const modelGuidance = noaa?.modelGuidance || {};

  const observedPoints = (observed?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  const officialPoints = (forecast?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  const observedSeries = observedPoints.length
    ? [{ label: 'Observed stage', style: 'observed', points: observedPoints.map((p) => ({ x: p.validTime, y: p.primary })) }]
    : [];

  const officialForecastSeries = officialPoints.length
    ? [{ label: 'Official NOAA forecast', style: 'forecast', points: officialPoints.map((p) => ({ x: p.validTime, y: p.primary })) }]
    : [];

  const nwmSeriesDefs = [
    ['analysisAssimilation', 'NWM analysis', 'history'],
    ['shortRange', 'NWM short-range', 'forecast'],
    ['mediumRange', 'NWM medium-range', 'secondary'],
    ['mediumRangeBlend', 'NWM medium blend', 'secondary'],
    ['longRange', 'NWM long-range', 'secondary']
  ];

  const modelSeries = nwmSeriesDefs.map(([key, label, style]) => {
    const series = modelGuidance?.[key];
    const points = (series?.data || []).filter((p) => Number.isFinite(p?.flow)).map((p) => ({ x: p.validTime, y: p.flow }));
    if (!points.length) return null;
    return {
      label,
      style,
      units: series.units,
      referenceTime: series.referenceTime,
      points
    };
  }).filter(Boolean);

  const primaryForecastSeries = officialForecastSeries.length
    ? officialForecastSeries
    : modelSeries.filter((series) => /short-range|medium-range|medium blend|long-range/i.test(series.label));

  const mainSeries = [
    ...observedSeries,
    ...primaryForecastSeries.slice(0, 3)
  ];

  const allSeries = [
    ...observedSeries,
    ...modelSeries,
    ...officialForecastSeries
  ];

  const crestSource = (officialForecastSeries[0]?.points || primaryForecastSeries[0]?.points || []).slice();
  let crest = null;
  for (const point of crestSource) {
    if (!crest || point.y > crest.y) crest = point;
  }

  const firstForecastPoint = (officialForecastSeries[0]?.points || primaryForecastSeries[0]?.points || [])[0] || null;
  const lastForecastPoint = (officialForecastSeries[0]?.points || primaryForecastSeries[0]?.points || []).at?.(-1) || null;
  const horizonDays = firstForecastPoint && lastForecastPoint
    ? Math.max(0, Math.round((new Date(lastForecastPoint.x) - new Date(firstForecastPoint.x)) / 86400000))
    : 0;

  let quality = 'Observed only';
  if (officialForecastSeries.length) quality = 'Official NOAA forecast';
  else if (primaryForecastSeries.length) quality = 'National Water Model guidance';
  else if (observedSeries.length) quality = 'Observed only';

  return {
    quality,
    issued: forecast?.issuedTime || modelSeries[1]?.referenceTime || modelSeries[0]?.referenceTime || null,
    nextValid: firstForecastPoint?.x || null,
    crest: crest?.y ?? null,
    crestTime: crest?.x ?? null,
    crestFlow: null,
    primaryUnits: forecast?.primaryUnits || primaryForecastSeries[0]?.units || observed?.primaryUnits || '',
    secondaryUnits: forecast?.secondaryUnits || '',
    horizonDays,
    observedSeries,
    forecastSeries: primaryForecastSeries,
    officialForecastSeries,
    modelSeries,
    series: allSeries,
    mainSeries
  };
}

async function loadStation(stationOrId) {
  if (typeof stationOrId === 'object' && stationOrId?.stationId) return stationOrId;
  const station = stations.find((item) => item.id === stationOrId || item.slug === stationOrId);
  if (!station) throw new Error('Unknown station');
  return fetchJsonWithFallback(`sensors/${encodeURIComponent(station.slug)}.json`);
}

async function showStation(stationOrId) {
  const baseStation = typeof stationOrId === 'object' ? stationOrId : stations.find((item) => item.id === stationOrId || item.slug === stationOrId);
  const loadingName = baseStation?.name || 'sensor';
  detailsEl.innerHTML = `<div class="card">Loading ${escapeHtml(loadingName)}…</div>`;
  try {
    const station = await loadStation(stationOrId);
    activeStationId = station.id;
    let body = '';
    if (station.country === 'US') {
      const [usgs, noaa] = await Promise.all([
        fetchUsgsSeries(station.stationId),
        station.noaaForecastPath ? fetchJsonWithFallback(station.noaaForecastPath.replace(/^data\//, '')).catch(() => null) : Promise.resolve(null)
      ]);
      const liveSeries = latestUsgsValue(usgs.iv);
      const dailySeries = latestUsgsValue(usgs.dv);
      const currentStageSeries = liveSeries.find((s) => /gage height/i.test(s.label));
      const currentFlowSeries = liveSeries.find((s) => /discharge/i.test(s.label));
      const currentStage = currentStageSeries?.points.at(-1)?.y;
      const currentFlow = currentFlowSeries?.points.at(-1)?.y;
      const forecast = summarizeForecast(noaa);
      const recentDailyStage = dailySeries.find((s) => /gage height/i.test(s.label));
      const recentDailyFlow = dailySeries.find((s) => /discharge/i.test(s.label));
      const usingModelGuidance = forecast.quality === 'National Water Model guidance';
      const primaryObservedSeries = usingModelGuidance
        ? (currentFlowSeries ? { ...currentFlowSeries, label: 'USGS current discharge', style: 'observed' } : null)
        : (currentStageSeries ? { ...currentStageSeries, label: 'USGS current stage', style: 'observed' } : null);
      const primaryHistorySeries = usingModelGuidance
        ? (recentDailyFlow ? { ...recentDailyFlow, label: 'Recent daily discharge', style: 'history' } : null)
        : (recentDailyStage ? { ...recentDailyStage, label: 'Recent daily stage', style: 'history' } : null);
      const syntheticHydrographForecast = !forecast.forecastSeries.length && !usingModelGuidance && primaryObservedSeries
        ? buildSyntheticForecastSeries(primaryObservedSeries)
        : null;
      const hydrographForecastSeries = [
        primaryHistorySeries,
        primaryObservedSeries,
        ...(forecast.forecastSeries.length ? forecast.forecastSeries.slice(0, 2) : []),
        ...(syntheticHydrographForecast ? [syntheticHydrographForecast] : [])
      ].filter(Boolean);
      const detailChartSeries = usingModelGuidance
        ? [primaryObservedSeries, ...forecast.forecastSeries.slice(0, 3)].filter(Boolean)
        : (hydrographForecastSeries.length ? hydrographForecastSeries : (forecast.mainSeries.length ? forecast.mainSeries : forecast.series));
      const overviewChartSeries = [
        primaryObservedSeries,
        ...forecast.forecastSeries.slice(0, 3),
        ...(syntheticHydrographForecast ? [syntheticHydrographForecast] : []),
        primaryHistorySeries
      ].filter(Boolean);
      const mlForecast = station.stationId ? await fetchJsonWithFallback(`ml/forecasts/${encodeURIComponent(station.stationId)}.json`).catch(() => null) : null;
      if (mlForecast) mlForecastByStationId.set(String(station.stationId), mlForecast);
      body = `
        <div class="station-title-row"><h2>${escapeHtml(station.name)}</h2>${mlForecast ? '<span class="forecast-chip"><span class="ml-badge">ML</span> Montana runoff forecast available</span>' : ''}</div>
        <div class="meta">USGS observed • ${station.state} • ${escapeHtml(station.stationId)}${station.noaaLid ? ` • NOAA ${escapeHtml(station.noaaLid)}` : ''}</div>
        <div class="kv">
          <div class="card"><b>Current stage</b><br>${currentStage ?? 'n/a'} ${escapeHtml(currentStageSeries?.unit || '')}</div>
          <div class="card"><b>Current discharge</b><br>${currentFlow ?? 'n/a'} ${escapeHtml(currentFlowSeries?.unit || '')}</div>
          <div class="card"><b>Best forecast</b><br>${escapeHtml(syntheticHydrographForecast ? 'Synthetic hydrograph forecast' : forecast.quality)}</div>
          <div class="card"><b>Forecast issued</b><br>${escapeHtml(formatDate(forecast.issued || syntheticHydrographForecast?.points?.[0]?.x || null))}</div>
          <div class="card"><b>Forecast crest</b><br>${forecast.crest ?? syntheticHydrographForecast?.points?.reduce((best, point) => (!best || point.y > best.y ? point : best), null)?.y ?? 'n/a'} ${escapeHtml(forecast.primaryUnits || currentStageSeries?.unit || '')}</div>
          <div class="card"><b>Crest time</b><br>${escapeHtml(formatDate(forecast.crestTime || syntheticHydrographForecast?.points?.reduce((best, point) => (!best || point.y > best.y ? point : best), null)?.x || null))}</div>
          <div class="card"><b>Forecast horizon</b><br>${forecast.horizonDays ? `~${forecast.horizonDays} days` : syntheticHydrographForecast ? '~1 day' : 'n/a'}</div>
        </div>
        <div class="detail-grid">
          ${renderSiteMapCard(station)}
          ${renderMiniChart(detailChartSeries, usingModelGuidance ? 'River flow guidance' : 'River forecast hydrograph', forecast.quality === 'Official NOAA forecast'
            ? (forecast.horizonDays ? `Official NOAA forecast with observed context and a clear now marker. Forecast extends roughly ${forecast.horizonDays} days.` : 'Official NOAA forecast with observed context and a clear now marker.')
            : forecast.quality === 'National Water Model guidance'
              ? 'No official NOAA stage forecast was available here, so this uses National Water Model flow guidance with discharge context.'
              : syntheticHydrographForecast
                ? 'No official hydrograph was available here, so this chart adds a short synthetic forecast based on the latest observed stage trend.'
                : 'Showing observed river conditions with a clear now marker.', { compact: true })}
        </div>
        <p><a href="https://waterdata.usgs.gov/monitoring-location/${encodeURIComponent(station.stationId)}/" target="_blank" rel="noreferrer">USGS station page</a>${station.noaaHydrographUrl ? ` • <a href="${station.noaaHydrographUrl}" target="_blank" rel="noreferrer">NOAA water forecast</a>` : ''}${station.noaaLid ? ` • <a href="https://water.noaa.gov/gauges/${encodeURIComponent(station.noaaLid)}" target="_blank" rel="noreferrer">water.noaa.gov gauge</a>` : ''}</p>
        ${renderMiniChart(overviewChartSeries, usingModelGuidance ? 'Easy-read flow guidance' : 'Hydrograph forecast', forecast.quality === 'Official NOAA forecast'
          ? 'Observed stage plus official forecast, with forecast shading and peak labeling.'
          : forecast.quality === 'National Water Model guidance'
            ? 'Recent discharge plus model guidance when official stage forecasts are unavailable.'
            : syntheticHydrographForecast
              ? 'Observed stage plus a short synthetic forecast generated from the latest trend.'
              : 'Recent stage with a clearer visual hydrograph and now marker.')}
        <p>Forecast source priority: <b>official NOAA / water.noaa.gov forecast</b> first, then <b>National Water Model guidance</b> when NOAA stage forecasts are empty, then a short <b>synthetic hydrograph forecast</b> from the recent observed trend. Click anywhere on the map to snap to the nearest river sensor.</p>
        ${mlForecast ? renderMlForecastCard(mlForecast) : ''}`;
    } else {
      const series = await fetchCanadaSeries(station.stationId);
      const latest = latestCanadaValues(series.realtime);
      const dailyPoints = (series.daily.features || []).map((f) => ({ x: f.properties.DATE, y: f.properties.LEVEL ?? f.properties.DISCHARGE })).filter((p) => p.y != null);
      const mlForecast = station.stationId ? await fetchJsonWithFallback(`ml/forecasts/${encodeURIComponent(station.stationId)}.json`).catch(() => null) : null;
      if (mlForecast) mlForecastByStationId.set(String(station.stationId), mlForecast);
      body = `
        <div class="station-title-row"><h2>${escapeHtml(station.name)}</h2>${mlForecast ? '<span class="forecast-chip"><span class="ml-badge">ML</span> Montana runoff forecast available</span>' : ''}</div>
        <div class="meta">Environment Canada observed • ${station.state} • ${escapeHtml(station.stationId)}</div>
        <div class="kv">
          <div class="card"><b>Current level</b><br>${latest.level ?? 'n/a'}</div>
          <div class="card"><b>Current discharge</b><br>${latest.discharge ?? 'n/a'}</div>
          <div class="card"><b>Historical daily mean</b><br>last 30 values</div>
          <div class="card"><b>Forecast quality</b><br>Observed only nationally</div>
        </div>
        <p><a href="https://wateroffice.ec.gc.ca/report/real_time_e.html?stn=${encodeURIComponent(station.stationId)}" target="_blank" rel="noreferrer">Environment Canada station page</a></p>
        ${renderMiniChart([
          { label: 'Realtime level', style: 'observed', points: latest.points },
          { label: 'Daily mean', style: 'history', points: dailyPoints }
        ], 'Observed hydrograph', 'Environment Canada realtime plus recent daily mean history with a clear now marker.')}`;
    }
    detailsEl.innerHTML = body;
    if (station.country === 'US') mountDetailMap(station);
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
mlForecastOnlyEl?.addEventListener('change', applyFilters);

try {
  startupStatus.phase = 'loading-index';
  const [stationData, sourceData, mlForecastData] = await Promise.all([
    fetchJsonWithFallback('sensors/index.json'),
    fetchJsonWithFallback('sources.json'),
    fetchJsonWithFallback('ml/forecasts/montana_runoff_forecast.json').catch(() => null)
  ]);
  if (!Array.isArray(stationData) || !stationData.length) {
    throw new Error('Station index loaded but was empty or invalid.');
  }
  mlForecastSummary = mlForecastData;
  if (mlForecastSummary?.stations?.length) {
    for (const item of mlForecastSummary.stations) {
      mlForecastByStationId.set(String(item.station_id), item);
    }
  }
  stations = stationData.map((station) => ({
    ...station,
    mlForecast: mlForecastByStationId.has(String(station.stationId || station.id || ''))
  }));
  filteredStations = stations;
  renderMarkers();
  startupStatus.phase = 'ready';
  summaryEl.innerHTML += `<div style="margin-top:8px;color:#9fb4c7">Built ${new Date(sourceData.generatedAt).toLocaleString()} • ${sourceData.usgsStations.toLocaleString()} USGS • ${sourceData.canadaStations.toLocaleString()} Canada • ${sourceData.matchedForecastStations.toLocaleString()} official NOAA forecast matches${mlForecastSummary?.stations_forecasted ? ` • ${mlForecastSummary.stations_forecasted.toLocaleString()} Montana ML forecast stations` : ''} • sensor details load on demand</div>`;
} catch (error) {
  console.error('Startup failed:', error);
  startupStatus.phase = 'error';
  const attempts = error?.attempts?.length ? `<div class="startup-attempts">${error.attempts.map((line) => `<div>${escapeHtml(line)}</div>`).join('')}</div>` : '';
  summaryEl.innerHTML = `<div class="startup-error"><b>Failed to load station index.</b><div class="mt8">${escapeHtml(String(error?.message || error))}</div>${attempts}</div>`;
  detailsEl.innerHTML = `<div class="card"><b>Startup failed.</b><br>The app could not load the station index, so the map cannot render sensors yet.</div>`;
}
