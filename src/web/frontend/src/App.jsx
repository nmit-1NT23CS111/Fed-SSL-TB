import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  BarChart3, 
  Cloud, 
  Database, 
  LayoutDashboard, 
  Microscope, 
  Network, 
  Settings, 
  ShieldCheck, 
  Upload,
  AlertCircle
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area 
} from 'recharts';

const API_BASE = "http://localhost:8000";

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [metrics, setMetrics] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error("Failed to fetch metrics", err);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    setIsUploading(true);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log("Prediction Result:", data);
      setPrediction(data);
    } catch (err) {
      console.error("Prediction failed", err);
    } finally {
      setIsUploading(false);
    }
  };

  const SidebarItem = ({ id, icon: Icon, label }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
        activeTab === id 
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
          : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
      }`}
    >
      <Icon size={20} />
      <span className="font-medium">{label}</span>
    </button>
  );

  const StatCard = ({ label, value, icon: Icon, trend, color }) => (
    <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl backdrop-blur-sm">
      <div className="flex justify-between items-start mb-4">
        <div className={`p-3 rounded-lg ${color} bg-opacity-10`}>
          <Icon className={color.replace('bg-', 'text-')} size={24} />
        </div>
        {trend && (
          <span className={`text-xs font-bold px-2 py-1 rounded-full bg-blue-500/10 text-blue-500`}>
            {trend}
          </span>
        )}
      </div>
      <div className="text-slate-400 text-sm font-medium">{label}</div>
      <div className="text-2xl font-bold text-white mt-1">{value}</div>
    </div>
  );

  const lastEval = metrics.findLast(m => m.eval_metrics);

  const currentLoss = metrics.length > 0 ? metrics[metrics.length - 1].mean_ssl_loss.toFixed(4) : "N/A";
  let lossTrend = null;
  let trendIsGood = true;
  if (metrics.length > 1) {
    const curr = metrics[metrics.length - 1].mean_ssl_loss;
    const prev = metrics[metrics.length - 2].mean_ssl_loss;
    const diff = ((prev - curr) / prev) * 100;
    lossTrend = `${diff > 0 ? '-' : '+'}${Math.abs(diff).toFixed(1)}%`;
    trendIsGood = diff > 0; // Negative loss growth is good
  }

  return (
    <div className="min-h-screen bg-[#05070a] text-slate-200 font-sans flex">
      <aside className="w-64 border-r border-slate-800 p-6 flex flex-col gap-8">
        <div className="flex items-center gap-3 px-2">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Activity className="text-white" size={24} />
          </div>
          <span className="text-xl font-bold tracking-tight text-white">FedSSL <span className="text-blue-500">AI</span></span>
        </div>
        <nav className="flex flex-col gap-2">
          <SidebarItem id="dashboard" icon={LayoutDashboard} label="Dashboard" />
          <SidebarItem id="analysis" icon={Microscope} label="TB Analysis" />
          <SidebarItem id="metrics" icon={BarChart3} label="Training Logs" />
          <SidebarItem id="federated" icon={Network} label="Federated Map" />
          <SidebarItem id="requirements" icon={ShieldCheck} label="Requirements" />
        </nav>
        <div className="mt-auto">
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-4 rounded-2xl border border-slate-700/50">
            <div className="text-sm font-bold text-white mb-1">RTX 2050 · CUDA Active</div>
            <div className="w-full bg-slate-700 h-1.5 rounded-full mt-3 overflow-hidden">
              <div className="bg-blue-500 w-full h-full"></div>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 p-8 overflow-y-auto">
        <header className="flex justify-between items-center mb-10">
          <div>
            <h1 className="text-3xl font-bold text-white capitalize">{activeTab.replace('_', ' ')}</h1>
            <p className="text-slate-500 mt-1">Federated Self-Supervised Learning for Tuberculosis Detection</p>
          </div>
          <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 border-2 border-slate-800"></div>
        </header>

        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard label="Global Loss" value={currentLoss} icon={Activity} trend={lossTrend} color={trendIsGood ? "bg-emerald-500" : "bg-red-500"} />
              <StatCard label="Best AUC" value={lastEval ? lastEval.eval_metrics.auc.toFixed(4) : "N/A"} icon={Network} color="bg-indigo-500" />
              <StatCard label="Total Samples" value={metrics.length > 0 ? `${(metrics[metrics.length - 1].sample_counts.reduce((a, b) => a + b, 0) / 1000).toFixed(0)}k / 112k` : "20k / 112k"} icon={Database} color="bg-blue-500" />
              <StatCard label="Round" value={`${metrics.length}/20`} icon={LayoutDashboard} color="bg-amber-500" />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-3xl p-8">
                <h3 className="text-xl font-bold text-white mb-6">Learning Curve (MAE Loss)</h3>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={metrics}>
                      <defs>
                        <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="round" stroke="#64748b" />
                      <YAxis stroke="#64748b" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }} />
                      <Area type="monotone" dataKey="mean_ssl_loss" stroke="#3b82f6" fillOpacity={1} fill="url(#colorLoss)" strokeWidth={3} name="Loss" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-start mb-6">
                    <h3 className="text-xl font-bold text-white">Few-Shot Progress</h3>
                    <span className="text-[10px] uppercase tracking-wider font-bold bg-blue-500/20 text-blue-400 px-2 py-1 rounded">5-Shot Inference</span>
                  </div>
                  {lastEval ? (
                    <div className="space-y-6">
                      <div className="text-5xl font-black text-blue-500">{(lastEval.eval_metrics.auc * 100).toFixed(1)}% <span className="text-sm text-slate-500 font-normal">AUC</span></div>
                      <div className="space-y-3">
                        <div className="flex justify-between text-sm"><span className="text-slate-400">Accuracy</span><span className="text-emerald-400 font-bold">{(lastEval.eval_metrics.accuracy * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between text-sm"><span className="text-slate-400">Sensitivity <span className="text-[10px] text-slate-600">(TB Caught)</span></span><span className="text-white font-bold">{(lastEval.eval_metrics.sensitivity * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between text-sm"><span className="text-slate-400">Specificity <span className="text-[10px] text-slate-600">(Healthy Caught)</span></span><span className="text-white font-bold">{(lastEval.eval_metrics.specificity * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between text-sm"><span className="text-slate-400">F1 Score</span><span className="text-white font-bold">{(lastEval.eval_metrics.f1 * 100).toFixed(1)}%</span></div>
                      </div>
                    </div>
                  ) : <p className="text-slate-600">Awaiting Round 5 Evaluation...</p>}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
            <div className="bg-slate-900/50 border-2 border-dashed border-slate-800 rounded-3xl p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
              {previewUrl ? <img src={previewUrl} className="max-h-[300px] rounded-xl mb-6 shadow-2xl" /> : <Upload size={48} className="text-blue-500 mb-4" />}
              <input type="file" onChange={handleFileUpload} className="hidden" id="fileIn" accept="image/*" />
              <label htmlFor="fileIn" className="bg-blue-600 px-8 py-3 rounded-xl font-bold cursor-pointer hover:bg-blue-500 transition-all">
                {previewUrl ? "Change Image" : "Upload Chest X-Ray"}
              </label>
            </div>
            <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 flex flex-col">
              <h3 className="text-xl font-bold text-white mb-8">Inference Result</h3>
              {isUploading ? <div className="flex-1 flex flex-col items-center justify-center animate-pulse text-blue-500 font-bold">Analyzing...</div> : 
               prediction ? (
                 <div className="space-y-8 animate-in fade-in duration-500">
                    <div>
                      <div className="text-slate-500 text-sm font-bold mb-1">DIAGNOSIS</div>
                      <div className={`text-5xl font-black ${prediction.prediction === 'TB Positive' ? 'text-red-500' : 'text-emerald-500'}`}>
                        {prediction.prediction}
                      </div>
                    </div>
                    <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50">
                      <div className="flex justify-between text-xs font-bold text-slate-400 mb-2">CONFIDENCE SCORE</div>
                      <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden">
                        <div className="bg-blue-500 h-full transition-all duration-1000" style={{ width: `${prediction.confidence * 100}%` }}></div>
                      </div>
                      <div className="mt-2 text-right text-white font-bold">{(prediction.confidence * 100).toFixed(1)}%</div>
                    </div>
                 </div>
               ) : <p className="text-slate-700 text-center my-auto">Upload an image to start analysis.</p>}
            </div>
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="space-y-8">
            <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8">
               <h3 className="text-xl font-bold text-white mb-8">Accuracy Improving Over Rounds</h3>
               <div className="h-[400px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metrics.filter(m => m.eval_metrics)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="round" stroke="#64748b" />
                      <YAxis domain={[0, 1]} stroke="#64748b" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }} />
                      <Line type="monotone" dataKey="eval_metrics.auc" stroke="#3b82f6" strokeWidth={4} name="AUC Score" dot={{ r: 6 }} />
                      <Line type="monotone" dataKey="eval_metrics.accuracy" stroke="#10b981" strokeWidth={2} name="Accuracy" />
                    </LineChart>
                  </ResponsiveContainer>
               </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
               <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8">
                  <h3 className="text-lg font-bold text-white mb-6">Hospital Performance (Loss)</h3>
                  <div className="space-y-4">
                    {metrics.length > 0 && metrics[metrics.length-1].hospital_losses.map((loss, idx) => (
                      <div key={idx} className="flex items-center gap-4">
                        <div className="text-xs font-bold text-slate-500 w-12">H{idx+1}</div>
                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden"><div className="bg-indigo-500 h-full" style={{ width: `${(1-loss)*100}%` }}></div></div>
                        <div className="text-xs font-bold text-white">{loss.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
               </div>
               <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 flex flex-col justify-center items-center">
                  <div className="text-blue-500 font-black text-4xl mb-2">{metrics.length}</div>
                  <div className="text-slate-400 font-bold uppercase tracking-widest text-xs">Communication Rounds Completed</div>
               </div>
            </div>
          </div>
        )}

        {activeTab === 'federated' && (
          <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-12 flex flex-col items-center justify-center min-h-[600px]">
             <div className="relative w-[500px] h-[500px] mb-12 flex items-center justify-center">
                {/* Central Cloud */}
                <div className="absolute h-24 w-24 bg-blue-600 rounded-full flex items-center justify-center shadow-[0_0_60px_rgba(37,99,235,0.6)] z-20 border-4 border-slate-900 animate-pulse">
                  <Cloud size={40} className="text-white" />
                </div>
                
                {/* Hospital Nodes and Data Lines */}
                {[0, 72, 144, 216, 288].map((a, i) => {
                  const radius = 200;
                  const top = `calc(50% + ${Math.sin(a*Math.PI/180)*radius}px - 28px)`;
                  const left = `calc(50% + ${Math.cos(a*Math.PI/180)*radius}px - 28px)`;
                  
                  return (
                    <React.Fragment key={i}>
                      {/* Connection Line */}
                      <div className="absolute top-1/2 left-1/2 w-full h-0.5 bg-slate-800 origin-left z-0" style={{
                        width: `${radius}px`,
                        transform: `rotate(${a}deg)`
                      }}>
                        {/* Animated Data Packet */}
                        <div className="absolute top-[-3px] left-0 w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,1)] z-10" style={{
                          animation: `ping-pong 3s ease-in-out infinite ${i * 0.4}s`
                        }}></div>
                      </div>
                      
                      {/* Hospital Node */}
                      <div className="absolute w-14 h-14 bg-slate-800 rounded-2xl border-2 border-slate-700 flex items-center justify-center z-10 transition-all hover:border-emerald-500 hover:scale-110 shadow-xl" style={{ top, left }}>
                        <Database size={24} className="text-emerald-500" />
                        <div className="absolute -bottom-8 text-xs font-bold text-slate-400 bg-slate-900 px-2 py-1 rounded-lg border border-slate-800">Hospital {i+1}</div>
                      </div>
                    </React.Fragment>
                  );
                })}
             </div>
             
             {/* Global Animation Styles */}
             <style dangerouslySetInnerHTML={{__html: `
               @keyframes ping-pong {
                 0% { left: 100%; opacity: 0; }
                 10% { opacity: 1; }
                 40% { left: 0%; opacity: 1; background-color: #3b82f6; box-shadow: 0 0 10px rgba(59,130,246,1); }
                 50% { left: 0%; opacity: 0; }
                 60% { left: 0%; opacity: 1; background-color: #10b981; box-shadow: 0 0 10px rgba(16,185,129,1); }
                 90% { left: 100%; opacity: 1; }
                 100% { left: 100%; opacity: 0; }
               }
             `}} />

             <div className="text-center max-w-2xl bg-slate-800/50 p-8 rounded-2xl border border-slate-700">
               <h3 className="text-3xl font-black text-white mb-4 flex items-center justify-center gap-3"><ShieldCheck className="text-blue-500"/> Privacy-First Collaboration</h3>
               <p className="text-slate-400 text-lg leading-relaxed">
                 Zero raw patient data leaves the hospital boundaries. Only mathematically encrypted model weights (the green packets) are sent to the central server. The global model aggregates this intelligence using the <span className="text-blue-400 font-bold">FedProx</span> protocol, and sends the upgraded "Brain" back to the hospitals (the blue packets).
               </p>
             </div>
          </div>
        )}

        {activeTab === 'requirements' && (
          <div className="space-y-8 animate-in fade-in duration-500">
            <div className="bg-gradient-to-r from-blue-900/50 to-indigo-900/50 border border-blue-500/30 rounded-3xl p-10 text-center">
              <h2 className="text-4xl font-black text-white mb-4">System Architecture & Specifications</h2>
              <p className="text-blue-300 text-lg">Technical backbone powering the Federated TB Detection Platform.</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:border-slate-600 transition-all">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-4 bg-slate-800 rounded-2xl"><Network className="text-blue-500" size={32}/></div>
                  <h3 className="text-2xl font-bold text-white">AI Architecture</h3>
                </div>
                <ul className="space-y-4">
                  <li className="flex justify-between items-center border-b border-slate-800 pb-3"><span className="text-slate-400">Backbone Model</span><span className="font-bold text-white">Vision Transformer (ViT-Tiny)</span></li>
                  <li className="flex justify-between items-center border-b border-slate-800 pb-3"><span className="text-slate-400">Pre-training Method</span><span className="font-bold text-white">Masked Autoencoder (MAE)</span></li>
                  <li className="flex justify-between items-center border-b border-slate-800 pb-3"><span className="text-slate-400">Federated Strategy</span><span className="font-bold text-white">FedProx (<span className="text-blue-400">μ = 0.01</span>)</span></li>
                  <li className="flex justify-between items-center pb-3"><span className="text-slate-400">Classification Head</span><span className="font-bold text-white">Prototypical Network (Few-Shot)</span></li>
                </ul>
              </div>

              <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:border-slate-600 transition-all">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-4 bg-slate-800 rounded-2xl"><Database className="text-emerald-500" size={32}/></div>
                  <h3 className="text-2xl font-bold text-white">Data Ecosystem</h3>
                </div>
                <ul className="space-y-4">
                  <li className="flex justify-between items-center border-b border-slate-800 pb-3"><span className="text-slate-400">Unsupervised Pre-training</span><span className="font-bold text-white">NIH Chest X-ray (20,000 imgs)</span></li>
                  <li className="flex justify-between items-center border-b border-slate-800 pb-3"><span className="text-slate-400">Few-Shot Support Set</span><span className="font-bold text-white">Shenzhen (5-Shot/Class)</span></li>
                  <li className="flex justify-between items-center border-b border-slate-800 pb-3"><span className="text-slate-400">Validation & Test Set</span><span className="font-bold text-white">Montgomery County (138 imgs)</span></li>
                  <li className="flex justify-between items-center pb-3"><span className="text-slate-400">Data Distribution Skew</span><span className="font-bold text-white">Dirichlet Non-IID (<span className="text-emerald-400">α = 2.0</span>)</span></li>
                </ul>
              </div>

              <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:border-slate-600 transition-all md:col-span-2">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-4 bg-slate-800 rounded-2xl"><Settings className="text-amber-500" size={32}/></div>
                  <h3 className="text-2xl font-bold text-white">Hardware & Performance Constraints</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700">
                    <div className="text-slate-400 text-sm font-bold mb-1">COMPUTE ENGINE</div>
                    <div className="text-xl font-black text-white">NVIDIA RTX 2050</div>
                    <div className="text-amber-500 text-sm mt-2">CUDA Accelerated</div>
                  </div>
                  <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700">
                    <div className="text-slate-400 text-sm font-bold mb-1">MEMORY CONSTRAINT</div>
                    <div className="text-xl font-black text-white">4 GB VRAM</div>
                    <div className="text-emerald-500 text-sm mt-2">Micro-batching (BS=4)</div>
                  </div>
                  <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700">
                    <div className="text-slate-400 text-sm font-bold mb-1">IMAGE RESOLUTION</div>
                    <div className="text-xl font-black text-white">224 x 224 pixels</div>
                    <div className="text-blue-500 text-sm mt-2">Bilinear Interpolation</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
