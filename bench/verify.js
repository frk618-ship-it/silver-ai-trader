// 驗證 index.html 內的 FusedBiGRU / FusedLSTM：
//   1) 與 TF.js 原生 Bidirectional(GRU) / LSTM 輸出、梯度一致（同權重）
//   2) save → load 往返一致
//   3) 訓練速度比較
// 用法：npm i @tensorflow/tfjs@4.22.0 @tensorflow/tfjs-backend-wasm@4.22.0 playwright && node verify.js [webgl]
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path'), http = require('http');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const a = html.indexOf('        class FusedBiGRU extends tf.layers.Layer');
const b = html.indexOf('try { tf.serialization.registerClass(FusedLSTM); } catch (_) {}');
if (a < 0 || b < 0) throw new Error('找不到融合層程式碼');
const layerSrc = html.slice(a, b) + 'try { tf.serialization.registerClass(FusedLSTM); } catch (_) {}\nwindow.FusedBiGRU=FusedBiGRU;window.FusedLSTM=FusedLSTM;';
const NM = path.join(__dirname, 'node_modules');
const page = `<!doctype html><script src="/tf.js"></script><script src="/wasm.js"></script><script>
tf.wasm.setWasmPaths('/wasm/');${layerSrc}
function mk(algo,lb,f,u,fused){const i=tf.input({shape:[lb,f]});let o;
 if(algo==='BiGRU'){ if(fused){o=new FusedBiGRU({units:u,returnSequences:true,recurrentInitializer:'glorotUniform'}).apply(i);o=tf.layers.dropout({rate:.3}).apply(o);o=new FusedBiGRU({units:u,recurrentInitializer:'glorotUniform'}).apply(o);}
  else{o=tf.layers.bidirectional({layer:tf.layers.gru({units:u,returnSequences:true,recurrentInitializer:'glorotUniform'})}).apply(i);o=tf.layers.dropout({rate:.3}).apply(o);o=tf.layers.bidirectional({layer:tf.layers.gru({units:u,recurrentInitializer:'glorotUniform'})}).apply(o);} }
 else { if(fused){o=new FusedLSTM({units:u,returnSequences:true}).apply(i);o=tf.layers.dropout({rate:.3}).apply(o);o=new FusedLSTM({units:u}).apply(o);}
  else{o=tf.layers.lstm({units:u,returnSequences:true}).apply(i);o=tf.layers.dropout({rate:.3}).apply(o);o=tf.layers.lstm({units:u}).apply(o);} }
 o=tf.layers.dense({units:1}).apply(o);return tf.model({inputs:i,outputs:o});}
window.check=async(be,algo)=>{await tf.setBackend(be);await tf.ready();
 const A=mk(algo,10,20,32,false),B=mk(algo,10,20,32,true);B.setWeights(A.getWeights());
 const X=tf.randomNormal([64,10,20]),Y=tf.randomNormal([64,1]);
 const pa=A.predict(X),pb=B.predict(X);const out=(await tf.max(tf.abs(tf.sub(pa,pb))).data())[0];
 const lf=m=>()=>tf.losses.meanSquaredError(Y,m.apply(X,{training:false}));
 const ga=tf.variableGrads(lf(A)).grads,gb=tf.variableGrads(lf(B)).grads;
 const ka=A.trainableWeights.map(w=>w.val),kb=B.trainableWeights.map(w=>w.val);let g=0;
 for(let i=0;i<ka.length;i++)g=Math.max(g,(await tf.max(tf.abs(tf.sub(ga[ka[i].name],gb[kb[i].name]))).data())[0]);
 await B.save('localstorage://verify');const C=await tf.loadLayersModel('localstorage://verify');
 const rt=(await tf.max(tf.abs(tf.sub(C.predict(X),pb))).data())[0];return {out,grad:g,roundTrip:rt};};
window.speed=async(be,algo,fused,bs)=>{await tf.setBackend(be);await tf.ready();
 const X=tf.randomUniform([480,10,20]),Y=tf.randomUniform([480,1]);const m=mk(algo,10,20,32,fused);
 m.compile({optimizer:tf.train.adam(1e-3),loss:(a,b)=>tf.losses.huberLoss(a,b)});
 await m.fit(X,Y,{epochs:1,batchSize:bs,yieldEvery:'never'});const t0=performance.now();
 await m.fit(X,Y,{epochs:3,batchSize:bs,yieldEvery:'never'});const ms=(performance.now()-t0)/3;tf.dispose([X,Y]);m.dispose();return ms;};
</script>`;
const files = { '/tf.js': path.join(NM, '@tensorflow/tfjs/dist/tf.min.js'), '/wasm.js': path.join(NM, '@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.min.js') };
const srv = http.createServer((q, r) => {
  if (q.url === '/') { r.writeHead(200, { 'Content-Type': 'text/html' }); return r.end(page); }
  const f = files[q.url] || (q.url.startsWith('/wasm/') ? path.join(NM, '@tensorflow/tfjs-backend-wasm/dist', q.url.slice(6)) : null);
  if (!f || !fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'Content-Type': f.endsWith('.wasm') ? 'application/wasm' : 'application/javascript' }); r.end(fs.readFileSync(f));
}).listen(0, async () => {
  const gl = process.argv.includes('webgl');
  const br = await chromium.launch({ args: gl ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] : [] });
  const pg = await br.newPage(); pg.on('pageerror', e => console.log('PAGE ERROR', e.message));
  await pg.goto(`http://localhost:${srv.address().port}/`);
  const backends = gl ? ['webgl', 'wasm', 'cpu'] : ['wasm', 'cpu'];
  let ok = true;
  for (const be of backends) for (const algo of ['BiGRU', 'LSTM']) {
    const r = await pg.evaluate(([be, a]) => check(be, a), [be, algo]);
    const pass = r.out < 1e-5 && r.grad < 1e-5 && r.roundTrip === 0; ok = ok && pass;
    console.log(`${pass ? 'PASS' : 'FAIL'} ${be.padEnd(5)} ${algo.padEnd(5)} 輸出差 ${r.out.toExponential(2)}  梯度差 ${r.grad.toExponential(2)}  存取往返 ${r.roundTrip}`);
  }
  for (const be of backends.filter(b => b !== 'cpu')) for (const algo of ['BiGRU', 'LSTM']) for (const bs of [32, 128]) {
    const n = await pg.evaluate(([be, a, bs]) => speed(be, a, false, bs), [be, algo, bs]);
    const f = await pg.evaluate(([be, a, bs]) => speed(be, a, true, bs), [be, algo, bs]);
    console.log(`速度 ${be.padEnd(5)} ${algo.padEnd(5)} batch ${String(bs).padEnd(3)} 原生 ${n.toFixed(0)}ms/epoch → 融合 ${f.toFixed(0)}ms/epoch (${(n / f).toFixed(2)}×)`);
  }
  await br.close(); srv.close(); process.exit(ok ? 0 : 1);
});
