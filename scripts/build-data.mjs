import { mkdir, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'generated');

const US_STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC','PR','VI','GU'];
const STATE_FIPS_TO_ABBR = {
  '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE','11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND','39':'OH','40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD','47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV','55':'WI','56':'WY','60':'AS','66':'GU','69':'MP','72':'PR','78':'VI'
};

async function fetchJson(url) {
  const res = await fetch(url, { headers: { 'user-agent': 'north-america-river-watch-builder' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url, { headers: { 'user-agent': 'north-america-river-watch-builder' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.text();
}

function parseRdb(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  const dataLines = lines.filter((line) => !line.startsWith('#'));
  if (dataLines.length < 3) return [];
  const headers = dataLines[0].split('\t');
  return dataLines.slice(2).map((line) => {
    const values = line.split('\t');
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
  });
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function titleCase(value = '') {
  return value.toLowerCase().replace(/\b\w/g, (m) => m.toUpperCase());
}

function mapUsgsSite(row) {
  const lat = num(row.dec_lat_va || row.dec_lat_va_); 
  const lon = num(row.dec_long_va || row.dec_long_va_);
  if (lat == null || lon == null) return null;
  return {
    id: `usgs:${row.site_no}`,
    key: row.site_no,
    network: 'USGS',
    country: 'US',
    stationId: row.site_no,
    name: row.station_nm,
    waterbody: row.station_nm,
    state: STATE_FIPS_TO_ABBR[row.state_cd] || row.state_cd,
    latitude: lat,
    longitude: lon,
    forecast: false,
    realtime: true,
    noaaLid: null,
    noaaForecast: false
  };
}

async function fetchUsgsStations() {
  const stations = [];
  for (const state of US_STATES) {
    const url = `https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded&siteStatus=all&hasDataTypeCd=iv&siteType=ST&stateCd=${state}`;
    try {
      const rows = parseRdb(await fetchText(url));
      for (const row of rows) {
        const mapped = mapUsgsSite(row);
        if (mapped) stations.push(mapped);
      }
      console.log(`USGS ${state}: ${rows.length}`);
    } catch (error) {
      console.warn(`USGS ${state} failed: ${error.message}`);
    }
  }
  return stations;
}

async function fetchCanadaStations() {
  const url = 'https://api.weather.gc.ca/collections/hydrometric-stations/items?f=json&limit=10000';
  const json = await fetchJson(url);
  return (json.features || []).map((feature) => ({
    id: `ca:${feature.properties.STATION_NUMBER}`,
    key: feature.properties.STATION_NUMBER,
    network: 'ECCC',
    country: 'CA',
    stationId: feature.properties.STATION_NUMBER,
    name: feature.properties.STATION_NAME,
    waterbody: feature.properties.STATION_NAME,
    state: feature.properties.PROV_TERR_STATE_LOC,
    latitude: feature.geometry.coordinates[1],
    longitude: feature.geometry.coordinates[0],
    forecast: false,
    realtime: !!feature.properties.REAL_TIME,
    noaaLid: null,
    noaaForecast: false,
    status: feature.properties.STATUS_EN
  }));
}

async function fetchNoaaGauges() {
  const features = [];
  let offset = 0;
  const pageSize = 2000;
  while (true) {
    const url = `https://mapservices.weather.noaa.gov/eventdriven/rest/services/water/riv_gauges/MapServer/0/query?where=1%3D1&outFields=*&returnGeometry=true&f=json&resultOffset=${offset}&resultRecordCount=${pageSize}`;
    const json = await fetchJson(url);
    const page = json.features || [];
    features.push(...page);
    console.log(`NOAA gauges fetched: ${features.length}`);
    if (page.length < pageSize) break;
    offset += pageSize;
  }
  return features.map((feature) => ({
    lid: feature.attributes.gaugelid,
    usgsId: feature.attributes.usgsid || null,
    name: feature.attributes.location,
    waterbody: feature.attributes.waterbody,
    state: feature.attributes.state,
    latitude: feature.geometry?.y,
    longitude: feature.geometry?.x,
    status: feature.attributes.status,
    action: feature.attributes.action,
    flood: feature.attributes.flood,
    moderate: feature.attributes.moderate,
    major: feature.attributes.major,
    units: feature.attributes.units,
    hydrographUrl: feature.attributes.url
  })).filter((item) => item.lid && item.latitude != null && item.longitude != null);
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

function joinStations(usgsStations, canadaStations, noaaGauges) {
  const stations = [...usgsStations, ...canadaStations];
  const usgsByState = new Map();
  for (const station of usgsStations) {
    if (!usgsByState.has(station.state)) usgsByState.set(station.state, []);
    usgsByState.get(station.state).push(station);
  }

  for (const gauge of noaaGauges) {
    const candidates = usgsByState.get(gauge.state) || [];
    let best = null;
    let bestDist = Infinity;
    for (const station of candidates) {
      const dist = haversineKm(gauge, station);
      if (dist < bestDist) {
        best = station;
        bestDist = dist;
      }
    }
    if (best && bestDist <= 1.5) {
      best.noaaLid = gauge.lid;
      best.noaaForecast = true;
      best.forecast = true;
      best.noaaWaterbody = gauge.waterbody;
      best.noaaStatus = gauge.status;
      best.noaaHydrographUrl = gauge.hydrographUrl;
    }
  }
  return stations;
}

async function fetchNoaaForecastFiles(stations) {
  const lids = [...new Set(stations.filter((s) => s.noaaLid).map((s) => s.noaaLid))];
  const forecastDir = path.join(outDir, 'noaa-forecast');
  await mkdir(forecastDir, { recursive: true });

  const concurrency = 8;
  let index = 0;

  async function worker() {
    while (index < lids.length) {
      const lid = lids[index++];
      const url = `https://api.water.noaa.gov/nwps/v1/gauges/${encodeURIComponent(lid)}/stageflow`;
      try {
        const json = await fetchJson(url);
        await writeFile(path.join(forecastDir, `${lid}.json`), JSON.stringify(json));
        console.log(`NOAA forecast ${lid}`);
      } catch (error) {
        console.warn(`NOAA forecast failed for ${lid}: ${error.message}`);
      }
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
}

async function main() {
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  const [usgsStations, canadaStations, noaaGauges] = await Promise.all([
    fetchUsgsStations(),
    fetchCanadaStations(),
    fetchNoaaGauges()
  ]);

  const stations = joinStations(usgsStations, canadaStations, noaaGauges);

  await writeFile(path.join(outDir, 'stations.json'), JSON.stringify(stations));
  const matchedForecastStations = stations.filter((s) => s.noaaLid).length;

  await writeFile(path.join(outDir, 'sources.json'), JSON.stringify({
    generatedAt: new Date().toISOString(),
    stations: stations.length,
    usgsStations: usgsStations.length,
    canadaStations: canadaStations.length,
    noaaGauges: noaaGauges.length,
    matchedForecastStations,
    notes: [
      'US current and historical series are fetched live from USGS Water Services from the browser.',
      'Canada real-time and historical daily mean series are fetched live from Environment and Climate Change Canada GeoMet APIs from the browser.',
      'NOAA NWPS forecast series are prefetched into static JSON because NOAA APIs do not expose permissive browser CORS headers.'
    ]
  }, null, 2));
  await fetchNoaaForecastFiles(stations);

  console.log(`Generated ${stations.length} stations`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
