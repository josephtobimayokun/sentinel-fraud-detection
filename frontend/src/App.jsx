import { useState, useEffect, useCallback, useRef } from "react";

const MERCHANTS = ["Amazon","Walmart","Target","Best Buy","Apple Store","Gas Station","Restaurant","Grocery","Pharmacy","Online Gaming"];
const CARD_TYPES = ["Visa","Mastercard","Amex","Discover"];
const COUNTRIES = ["US","CA","GB","DE","FR","AU","JP","BR","IN","MX"];
const API = "http://localhost:8000";

const randId = () => "TXN-" + Math.random().toString(36).slice(2,10).toUpperCase();
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const randFloat = (lo, hi) => +(Math.random() * (hi - lo) + lo).toFixed(2);
const randInt = (lo, hi) => Math.floor(Math.random() * (hi - lo + 1)) + lo;

function genTx() {
  return {
    TransactionID: randId(),
    TransactionAmt: randFloat(10, 5000),
    Merchant: pick(MERCHANTS),
    CardType: pick(CARD_TYPES),
    Country: pick(COUNTRIES),
    Hour: randInt(0, 23),
    Velocity: randInt(0, 10),
    timestamp: new Date().toISOString(),
  };
}

function localScore(tx) {
  const hrm = ["Online Gaming","Apple Store","Best Buy"];
  const hrc = ["BR","MX","IN"];
  let s = Math.random() * 0.35;
  if (tx.Velocity > 7) s += 0.28;
  if (tx.TransactionAmt > 2000) s += 0.18;
  if (tx.Hour < 6 || tx.Hour > 22) s += 0.12;
  if (hrm.includes(tx.Merchant)) s += 0.10;
  if (hrc.includes(tx.Country)) s += 0.10;
  s = Math.min(0.99, Math.max(0.01, s + (Math.random()-0.5)*0.12));
  const impacts = [
    { feature:"Velocity", impact: tx.Velocity>7?0.38:0.08, value: tx.Velocity },
    { feature:"Amount",   impact: tx.TransactionAmt>2000?0.30:0.12, value: tx.TransactionAmt },
    { feature:"Hour",     impact: (tx.Hour<6||tx.Hour>22)?0.18:0.06, value: tx.Hour },
    { feature:"Country",  impact: hrc.includes(tx.Country)?0.15:0.04, value: tx.Country },
    { feature:"Merchant", impact: hrm.includes(tx.Merchant)?0.10:0.03, value: tx.Merchant },
  ].sort((a,b)=>b.impact-a.impact);
  const tot = impacts.reduce((s,f)=>s+f.impact,0);
  impacts.forEach(f=>f.impact=+(f.impact/tot).toFixed(3));
  return { TransactionID:tx.TransactionID, fraud_probability:+s.toFixed(4), prediction:s>0.5?"FRAUD":"LEGITIMATE", top_features:impacts };
}

const riskColor = p => p>=0.7?"#ef4444":p>=0.3?"#f59e0b":"#22c55e";
const fmtAmt = v => "$"+Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const fmtTime = iso => { try{return new Date(iso).toLocaleTimeString();}catch{return "";} };

function LiveDot({active}){
  return(
    <span style={{display:"inline-flex",alignItems:"center",gap:6}}>
      <span style={{width:8,height:8,borderRadius:"50%",background:active?"#22c55e":"#64748b",
        boxShadow:active?"0 0 0 3px rgba(34,197,94,0.25)":"none",
        animation:active?"sentPulse 1.5s infinite":"none",display:"inline-block"}}/>
      <span style={{fontSize:11,color:active?"#22c55e":"#64748b",letterSpacing:1,fontFamily:"monospace"}}>
        {active?"LIVE":"PAUSED"}
      </span>
    </span>
  );
}

function RiskGauge({probability}){
  const pct=Math.round(probability*100);
  const col=riskColor(probability);
  const deg=pct*3.6;
  const greenEnd=Math.min(pct,30)*3.6;
  const yellowEnd=Math.min(pct,70)*3.6;
  const conicGrad=`conic-gradient(
    #22c55e 0deg ${greenEnd}deg,
    #f59e0b ${greenEnd}deg ${yellowEnd}deg,
    #ef4444 ${yellowEnd}deg ${deg}deg,
    #0f172a ${deg}deg 360deg
  )`;
  return(
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:10}}>
      <div style={{width:140,height:140,borderRadius:"50%",background:conicGrad,display:"flex",
        alignItems:"center",justifyContent:"center",boxShadow:`0 0 30px ${col}40`}}>
        <div style={{width:102,height:102,borderRadius:"50%",background:"#1e293b",
          display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
          <span style={{fontSize:30,fontWeight:700,color:col,fontFamily:"monospace",lineHeight:1}}>{pct}%</span>
          <span style={{fontSize:9,color:"#94a3b8",letterSpacing:2,marginTop:2}}>FRAUD RISK</span>
        </div>
      </div>
      <div style={{padding:"4px 14px",borderRadius:4,background:`${col}20`,
        border:`1px solid ${col}60`,color:col,fontFamily:"monospace",fontSize:11,letterSpacing:2}}>
        {probability>=0.5?"⚠ FRAUD DETECTED":"✓ LEGITIMATE"}
      </div>
    </div>
  );
}

function FeatureBar({feature,impact,value}){
  const pct=Math.round(impact*100);
  const col=impact>0.3?"#ef4444":impact>0.15?"#f59e0b":"#22c55e";
  const displayVal=typeof value==="number"?(value>100?fmtAmt(value):value):value;
  return(
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4,fontSize:11}}>
        <span style={{color:"#94a3b8"}}>{feature}</span>
        <span style={{fontFamily:"monospace",color:"#e2e8f0"}}>
          {displayVal}
          <span style={{color:col,marginLeft:8}}>↑{pct}%</span>
        </span>
      </div>
      <div style={{height:5,borderRadius:3,background:"#0f172a",overflow:"hidden"}}>
        <div style={{height:"100%",width:`${pct}%`,background:`linear-gradient(90deg,${col}80,${col})`,
          borderRadius:3,transition:"width 0.6s ease"}}/>
      </div>
    </div>
  );
}

function MetricCard({label,value,color}){
  return(
    <div style={{background:"#0f172a",borderRadius:10,padding:"12px 14px",
      border:`1px solid ${color||"#334155"}`,boxShadow:color?`0 0 12px ${color}22`:"none"}}>
      <div style={{fontSize:22,fontWeight:700,fontFamily:"monospace",color:color||"#e2e8f0"}}>{value}</div>
      <div style={{fontSize:10,color:"#64748b",marginTop:2,letterSpacing:1}}>{label}</div>
    </div>
  );
}

function Histogram({transactions}){
  const buckets=[0,0,0,0,0];
  transactions.forEach(tx=>{
    if(tx.fraud_probability==null)return;
    buckets[Math.min(4,Math.floor(tx.fraud_probability*5))]++;
  });
  const max=Math.max(...buckets,1);
  const labels=["0–20%","20–40%","40–60%","60–80%","80–100%"];
  const colors=["#22c55e","#84cc16","#f59e0b","#f97316","#ef4444"];
  return(
    <div>
      <div style={{fontSize:10,color:"#64748b",marginBottom:10,letterSpacing:2}}>SCORE DISTRIBUTION</div>
      <div style={{display:"flex",gap:5,alignItems:"flex-end",height:70}}>
        {buckets.map((count,i)=>(
          <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:3}}>
            <span style={{fontSize:9,color:"#64748b",fontFamily:"monospace"}}>{count||""}</span>
            <div style={{width:"100%",borderRadius:"3px 3px 0 0",background:colors[i],
              height:`${(count/max)*55}px`,minHeight:count>0?3:0,transition:"height 0.4s ease",opacity:0.85}}/>
          </div>
        ))}
      </div>
      <div style={{display:"flex",gap:5,marginTop:4}}>
        {labels.map((l,i)=>(
          <div key={i} style={{flex:1,fontSize:8,color:"#475569",textAlign:"center"}}>{l}</div>
        ))}
      </div>
    </div>
  );
}

export default function FraudDashboard(){
  const [transactions,setTransactions]=useState([]);
  const [selectedTx,setSelectedTx]=useState(null);
  const [decisions,setDecisions]=useState({});
  const [isPaused,setIsPaused]=useState(false);
  const [note,setNote]=useState("");
  const [backendAlive,setBackendAlive]=useState(null);
  const [actionLoading,setActionLoading]=useState(false);
  const listRef=useRef(null);
  const intervalRef=useRef(null);

  useEffect(()=>{
    fetch(`${API}/health`,{signal:AbortSignal.timeout(2000)})
      .then(()=>setBackendAlive(true)).catch(()=>setBackendAlive(false));
  },[]);

  const generate=useCallback(async()=>{
    const tx=genTx();
    let scored;
    try{
      const res=await fetch(`${API}/score`,{method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify(tx),
        signal:AbortSignal.timeout(3000)});
      if(!res.ok)throw new Error();
      scored=await res.json();
      setBackendAlive(true);
    }catch{
      scored=localScore(tx);
      setBackendAlive(false);
    }
    const full={...tx,...scored};
    setTransactions(prev=>[full,...prev].slice(0,100));
    if(listRef.current)listRef.current.scrollTop=0;
  },[]);

  useEffect(()=>{
    if(isPaused){clearInterval(intervalRef.current);}
    else{
      generate();
      intervalRef.current=setInterval(generate,3000);
    }
    return()=>clearInterval(intervalRef.current);
  },[isPaused,generate]);

  const reviewed=Object.keys(decisions).length;
  const fraudCaught=Object.values(decisions).filter(d=>d.action==="BLOCK"&&d.modelPrediction==="FRAUD").length;
  const falsePositives=Object.values(decisions).filter(d=>d.action==="APPROVE"&&d.modelPrediction==="FRAUD").length;
  const fraudRate=transactions.length?(transactions.filter(t=>t.prediction==="FRAUD").length/transactions.length*100).toFixed(1):0;
  const recentDecisions=Object.values(decisions).sort((a,b)=>new Date(b.timestamp)-new Date(a.timestamp)).slice(0,5);

  async function handleAction(action){
    if(!selectedTx)return;
    setActionLoading(true);
    const payload={transaction_id:selectedTx.TransactionID,action,note:note||"",
      modelPrediction:selectedTx.prediction,modelScore:selectedTx.fraud_probability,
      timestamp:new Date().toISOString()};
    try{
      await fetch(`${API}/decision`,{method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),
        signal:AbortSignal.timeout(3000)});
    }catch{}
    setDecisions(prev=>({...prev,[selectedTx.TransactionID]:payload}));
    setSelectedTx(null);setNote("");setActionLoading(false);
  }

  const panel={background:"#1e293b",borderRadius:12,padding:16,border:"1px solid #334155",
    overflow:"hidden",display:"flex",flexDirection:"column"};

  return(
    <div style={{minHeight:"100vh",background:"#0f172a",color:"#e2e8f0",
      fontFamily:"'IBM Plex Mono','Courier New',monospace",padding:16,boxSizing:"border-box"}}>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');
        @keyframes sentPulse{0%,100%{box-shadow:0 0 0 3px rgba(34,197,94,0.25)}50%{box-shadow:0 0 0 7px rgba(34,197,94,0.05)}}
        @keyframes sentFadeIn{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:3px}
        ::-webkit-scrollbar-track{background:#0f172a}
        ::-webkit-scrollbar-thumb{background:#334155;border-radius:2px}
        .sent-tx:hover{background:#243447 !important;cursor:pointer}
        .sent-btn:hover{filter:brightness(1.25);transform:translateY(-1px)}
        .sent-btn:active{transform:translateY(0)}
        @media(max-width:900px){.sent-grid{grid-template-columns:1fr !important;height:auto !important}
          .sent-stream{max-height:280px}}
      `}</style>

      {/* Header */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",
        marginBottom:14,flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:34,height:34,borderRadius:8,fontSize:18,
            background:"linear-gradient(135deg,#ef4444 0%,#7c3aed 100%)",
            display:"flex",alignItems:"center",justifyContent:"center"}}>⚡</div>
          <div>
            <div style={{fontSize:15,fontWeight:700,letterSpacing:3,color:"#e2e8f0"}}>FRAUD</div>
            <div style={{fontSize:8,color:"#475569",letterSpacing:3}}>DETECTION SYSTEM</div>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:10,flexWrap:"wrap"}}>
          {backendAlive!==null&&(
            <span style={{fontSize:9,letterSpacing:1,
              color:backendAlive?"#22c55e":"#f59e0b",
              background:backendAlive?"#22c55e18":"#f59e0b18",
              border:`1px solid ${backendAlive?"#22c55e44":"#f59e0b44"}`,
              padding:"3px 8px",borderRadius:4,fontFamily:"monospace"}}>
              {backendAlive?"● BACKEND":"◌ LOCAL FALLBACK"}
            </span>
          )}
          <LiveDot active={!isPaused}/>
          <button onClick={()=>setIsPaused(p=>!p)} style={{
            padding:"6px 14px",borderRadius:6,cursor:"pointer",fontFamily:"monospace",fontSize:11,letterSpacing:1,
            background:isPaused?"#22c55e18":"#ef444418",
            border:`1px solid ${isPaused?"#22c55e44":"#ef444444"}`,
            color:isPaused?"#22c55e":"#ef4444"}}>
            {isPaused?"▶ RESUME":"⏸ PAUSE"}
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="sent-grid" style={{display:"grid",gridTemplateColumns:"320px 1fr 280px",
        gap:12,height:"calc(100vh - 92px)"}}>

        {/* LEFT: Stream */}
        <div className="sent-stream" style={{...panel}}>
          <div style={{fontSize:9,color:"#64748b",letterSpacing:2,marginBottom:10,display:"flex",justifyContent:"space-between"}}>
            <span>LIVE STREAM</span>
            <span style={{fontFamily:"monospace"}}>{transactions.length}/100</span>
          </div>
          <div ref={listRef} style={{overflowY:"auto",flex:1}}>
            {transactions.map(tx=>{
              const isSel=selectedTx?.TransactionID===tx.TransactionID;
              const isRev=!!decisions[tx.TransactionID];
              const col=riskColor(tx.fraud_probability);
              return(
                <div key={tx.TransactionID} className="sent-tx"
                  onClick={()=>{setSelectedTx(tx);setNote("");}}
                  style={{padding:"10px 11px",borderRadius:8,marginBottom:6,
                    background:isSel?"#0f172a":"transparent",
                    border:`1px solid ${isSel?col:"#334155"}`,
                    opacity:isRev?0.4:1,transition:"all 0.2s",
                    animation:"sentFadeIn 0.35s ease",
                    boxShadow:isSel?`0 0 14px ${col}30`:"none"}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4}}>
                    <span style={{fontSize:9,color:"#64748b",letterSpacing:1,fontFamily:"monospace"}}>
                      {tx.TransactionID}
                    </span>
                    <span style={{fontSize:8,padding:"2px 7px",borderRadius:3,letterSpacing:1,
                      background:`${col}22`,border:`1px solid ${col}55`,color:col}}>
                      {tx.prediction==="FRAUD"?"⚠ FRAUD":"✓ LEGIT"}
                    </span>
                  </div>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                    <div>
                      <div style={{fontSize:13,fontWeight:600,color:"#e2e8f0"}}>{fmtAmt(tx.TransactionAmt)}</div>
                      <div style={{fontSize:10,color:"#64748b",marginTop:2}}>{tx.Merchant} · {tx.Country}</div>
                    </div>
                    <div style={{textAlign:"right"}}>
                      <div style={{fontSize:20,fontWeight:700,color:col,fontFamily:"monospace",lineHeight:1}}>
                        {Math.round(tx.fraud_probability*100)}%
                      </div>
                      <div style={{fontSize:9,color:"#475569",marginTop:2}}>{fmtTime(tx.timestamp)}</div>
                    </div>
                  </div>
                  <div style={{height:2,borderRadius:1,background:"#0f172a",marginTop:8,overflow:"hidden"}}>
                    <div style={{height:"100%",width:`${tx.fraud_probability*100}%`,background:col,borderRadius:1}}/>
                  </div>
                </div>
              );
            })}
            {transactions.length===0&&(
              <div style={{textAlign:"center",color:"#334155",fontSize:12,marginTop:50,lineHeight:2}}>
                <div style={{fontSize:28,opacity:0.3}}>⚡</div>
                Waiting for transactions…
              </div>
            )}
          </div>
        </div>

        {/* CENTER: Detail */}
        <div style={{...panel}}>
          {selectedTx?(
            <>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:14}}>
                <div>
                  <div style={{fontSize:9,color:"#64748b",letterSpacing:2}}>TRANSACTION REVIEW</div>
                  <div style={{fontFamily:"monospace",fontSize:12,color:"#94a3b8",marginTop:2}}>
                    {selectedTx.TransactionID}
                  </div>
                </div>
                <button onClick={()=>setSelectedTx(null)} style={{background:"none",border:"none",
                  color:"#475569",cursor:"pointer",fontSize:18,padding:0,lineHeight:1}}>✕</button>
              </div>

              <div style={{display:"flex",justifyContent:"center",marginBottom:18}}>
                <RiskGauge probability={selectedTx.fraud_probability}/>
              </div>

              <div style={{marginBottom:14}}>
                <div style={{fontSize:9,color:"#64748b",letterSpacing:2,marginBottom:10}}>TOP RISK FACTORS</div>
                {(selectedTx.top_features||[]).map((f,i)=>(
                  <FeatureBar key={i} feature={f.feature} impact={f.impact} value={f.value}/>
                ))}
              </div>

              <div style={{marginBottom:14}}>
                <div style={{fontSize:9,color:"#64748b",letterSpacing:2,marginBottom:10}}>RAW DATA</div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
                  {[["Amount",fmtAmt(selectedTx.TransactionAmt)],["Merchant",selectedTx.Merchant],
                    ["Card",selectedTx.CardType],["Country",selectedTx.Country],
                    ["Hour",`${selectedTx.Hour}:00`],["Velocity",`${selectedTx.Velocity}/hr`]
                  ].map(([k,v])=>(
                    <div key={k} style={{background:"#0f172a",borderRadius:6,padding:"8px 10px",
                      border:"1px solid #334155"}}>
                      <div style={{fontSize:8,color:"#475569",letterSpacing:1,marginBottom:2}}>{k}</div>
                      <div style={{fontSize:12,fontFamily:"monospace",color:"#e2e8f0"}}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>

              <textarea value={note} onChange={e=>setNote(e.target.value)}
                placeholder="Analyst notes (optional)…" rows={2}
                style={{width:"100%",marginBottom:12,padding:"8px 10px",
                  background:"#0f172a",border:"1px solid #334155",borderRadius:6,
                  color:"#e2e8f0",fontFamily:"monospace",fontSize:11,resize:"vertical",outline:"none"}}/>

              <div style={{display:"flex",gap:8}}>
                {[["✓ APPROVE","#22c55e","APPROVE"],["⚑ 3DS","#f59e0b","3DS"],["✕ BLOCK","#ef4444","BLOCK"]].map(([label,col,action])=>(
                  <button key={action} className="sent-btn" disabled={actionLoading}
                    onClick={()=>handleAction(action)}
                    style={{flex:1,padding:"10px 0",borderRadius:8,border:`1px solid ${col}60`,
                      background:`${col}18`,color:col,cursor:actionLoading?"not-allowed":"pointer",
                      fontFamily:"monospace",fontSize:11,letterSpacing:1,transition:"all 0.2s",
                      opacity:actionLoading?0.5:1}}>
                    {label}
                  </button>
                ))}
              </div>
            </>
          ):(
            <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",
              justifyContent:"center",gap:12}}>
              <div style={{fontSize:48,opacity:0.1}}>⚡</div>
              <div style={{fontSize:12,color:"#334155",textAlign:"center",lineHeight:2}}>
                Select a transaction<br/>from the stream to review
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: Metrics */}
        <div style={{...panel,gap:12}}>
          <div style={{fontSize:9,color:"#64748b",letterSpacing:2}}>ANALYST METRICS</div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
            <MetricCard label="REVIEWED" value={reviewed} color="#7c3aed"/>
            <MetricCard label="FRAUD CAUGHT" value={fraudCaught} color="#ef4444"/>
            <MetricCard label="FALSE POS." value={falsePositives} color="#f59e0b"/>
            <MetricCard label="FRAUD RATE" value={`${fraudRate}%`} color={fraudRate>20?"#ef4444":"#22c55e"}/>
          </div>

          <div style={{background:"#0f172a",borderRadius:10,padding:14,border:"1px solid #334155"}}>
            <Histogram transactions={transactions}/>
          </div>

          <div style={{flex:1,overflow:"hidden",display:"flex",flexDirection:"column"}}>
            <div style={{fontSize:9,color:"#64748b",letterSpacing:2,marginBottom:8}}>RECENT DECISIONS</div>
            <div style={{overflowY:"auto",flex:1}}>
              {recentDecisions.length===0?(
                <div style={{fontSize:11,color:"#334155",textAlign:"center",marginTop:20}}>
                  No decisions logged
                </div>
              ):recentDecisions.map((d,i)=>{
                const col=d.action==="BLOCK"?"#ef4444":d.action==="3DS"?"#f59e0b":"#22c55e";
                return(
                  <div key={i} style={{padding:"8px 10px",marginBottom:6,borderRadius:6,
                    background:"#0f172a",border:"1px solid #334155",animation:"sentFadeIn 0.3s ease"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                      <span style={{fontSize:9,fontFamily:"monospace",color:"#64748b"}}>
                        {d.transaction_id?.slice(0,13)}
                      </span>
                      <span style={{fontSize:9,padding:"1px 6px",borderRadius:3,letterSpacing:1,
                        background:`${col}22`,color:col,border:`1px solid ${col}44`}}>{d.action}</span>
                    </div>
                    <div style={{fontSize:10,color:"#475569",marginTop:3}}>
                      {Math.round(d.modelScore*100)}% · {d.modelPrediction}
                    </div>
                    {d.note&&<div style={{fontSize:9,color:"#334155",marginTop:2,fontStyle:"italic",
                      overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{d.note}</div>}
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{borderTop:"1px solid #1e293b",paddingTop:10,
            fontSize:8,color:"#1e293b",fontFamily:"monospace",letterSpacing:1,
            background:"linear-gradient(transparent,#1e293b)",
            color:"#334155"}}>
            Joseph Tobi Mayokun v1.0 · PYTORCH MLP · {transactions.length} TXN
          </div>
        </div>
      </div>
    </div>
  );
}
