import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { webcrypto } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const backend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = fs.readFileSync(path.join(backend, 'static/js/pages/dredgingForecast.js'), 'utf8');
const code = source.slice(source.indexOf('  async function forecastHashMatches('), source.indexOf('  function updateComparison()'));
const manifest = JSON.parse(fs.readFileSync(path.join(backend, 'static/data/anchorage/manifest.json')));
const comparisons = JSON.parse(fs.readFileSync(path.join(backend, 'static/data/anchorage/comparisons.json')));

function harness(records = comparisons) {
  const context = { crypto: webcrypto, TextEncoder, state: { comparisons: records }, prepareCollection: collection => ({ collection }) };
  vm.createContext(context);
  vm.runInContext(`${code}\nglobalThis.attach = attachComparisons;`, context);
  return context;
}

test('all model/date comparisons attach with Linux and Windows line endings', async () => {
  const context = harness();
  let checked = 0;
  for (const frame of manifest.frames) {
    for (const [id, url] of Object.entries(frame.outputs)) {
      const raw = fs.readFileSync(path.join(backend, url), 'utf8').replace(/\r\n/g, '\n');
      for (const text of new Set([raw, raw.replace(/\n/g, '\r\n')])) {
        const prepared = { collection: JSON.parse(text) };
        await context.attach(prepared, frame, { id }, url, text);
        assert.ok(prepared.comparison?.count > 0, `${id} ${frame.id}: missing comparison`);
        assert.ok(Number.isFinite(prepared.comparison.mae));
        assert.ok(Number.isFinite(prepared.comparison.baseline));
        const cell = prepared.collection.features.find(f => Number.isFinite(f.properties._origin_depth_ft)).properties;
        assert.equal(cell._baseline_error_ft, cell._origin_depth_ft - cell._observed_depth_ft);
        assert.equal(cell._error_ft, cell._predicted_depth_ft - cell._observed_depth_ft);
        checked++;
      }
    }
  }
  assert.ok(checked >= 18);
});

test('modified prediction values remain rejected', async () => {
  const frame = manifest.frames[0], id = 'xgboost', url = frame.outputs[id];
  const collection = JSON.parse(fs.readFileSync(path.join(backend, url), 'utf8'));
  collection.features[0].properties.depth_ft += 1;
  const raw = JSON.stringify(collection), prepared = { collection };
  await harness().attach(prepared, frame, { id }, url, raw);
  assert.equal(prepared.comparison, undefined);
});

test('incomplete observation arrays remain rejected', async () => {
  const frame = manifest.frames[0], id = 'xgboost', url = frame.outputs[id];
  const records = structuredClone(comparisons);
  records.frames[0].outputs[id].origin_target_depths_ft.pop();
  const raw = fs.readFileSync(path.join(backend, url), 'utf8'), prepared = { collection: JSON.parse(raw) };
  await harness(records).attach(prepared, frame, { id }, url, raw);
  assert.equal(prepared.comparison, undefined);
});
