"use client";

import React, { useState, useEffect, useCallback } from "react";
import { generateMarketBrief, getAgentsStatus, getConnections, createConnection, deleteConnection, testConnection, MarketBriefResponse, ApiConnection } from "@/lib/api";
import { Mic, ArrowRight, Activity, TrendingUp, AlertCircle, ShieldAlert, Bot, MessageSquare, Plus, X, RefreshCw, Trash2, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const mockChartData = [
  { time: "09:30", value: 102.3 },
  { time: "10:30", value: 103.1 },
  { time: "11:30", value: 102.8 },
  { time: "12:30", value: 104.5 },
  { time: "13:30", value: 104.2 },
  { time: "14:30", value: 105.7 },
  { time: "15:30", value: 106.1 },
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<MarketBriefResponse | null>(null);
  const [error, setError] = useState("");
  const [healthStatus, setHealthStatus] = useState<{ status: string } | null>(null);

  // Connections state
  const [connections, setConnections] = useState<ApiConnection[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [addForm, setAddForm] = useState({ name: "", provider: "openai", api_key: "", base_url: "" });
  const [addLoading, setAddLoading] = useState(false);
  const [addError, setAddError] = useState("");
  const [testingId, setTestingId] = useState<string | null>(null);

  const fetchConnections = useCallback(() => {
    getConnections()
      .then((data) => setConnections(data.connections))
      .catch(() => setConnections([]));
  }, []);

  useEffect(() => {
    getAgentsStatus()
      .then(setHealthStatus)
      .catch((err) => console.error("Failed to fetch health status:", err));
    fetchConnections();
  }, [fetchConnections]);

  const handleAddConnection = async () => {
    if (!addForm.name.trim() || !addForm.api_key.trim()) return;
    setAddLoading(true);
    setAddError("");
    try {
      await createConnection({
        name: addForm.name,
        provider: addForm.provider,
        api_key: addForm.api_key,
        base_url: addForm.base_url || undefined,
      });
      setShowAddModal(false);
      setAddForm({ name: "", provider: "openai", api_key: "", base_url: "" });
      fetchConnections();
    } catch (err) {
      setAddError(err instanceof Error ? err.message : "Failed to add connection.");
    } finally {
      setAddLoading(false);
    }
  };

  const handleDeleteConnection = async (id: string) => {
    try {
      await deleteConnection(id);
      fetchConnections();
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const handleTestConnection = async (id: string) => {
    setTestingId(id);
    try {
      await testConnection(id);
      fetchConnections();
    } catch (err) {
      console.error("Test failed:", err);
    } finally {
      setTestingId(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");

    try {
      const res = await generateMarketBrief({ query });
      setResponse(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex-1 p-8 overflow-y-auto">
      <header className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Market Overview</h2>
          <p className="text-slate-500 mt-1">Real-time insights powered by FinAgent OS</p>
        </div>
        {healthStatus && (
          <div className="flex items-center space-x-2 px-4 py-2 bg-white rounded-full shadow-sm border border-slate-100 text-sm">
            <span className={`w-2.5 h-2.5 rounded-full ${healthStatus.status === "healthy" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"}`}></span>
            <span className="font-medium text-slate-700 capitalize">System {healthStatus.status}</span>
          </div>
        )}
      </header>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {[
          { label: "Portfolio Value", value: "$1.24M", change: "+2.4%", icon: TrendingUp, color: "text-emerald-600", bg: "bg-emerald-100" },
          { label: "S&P 500", value: "5,102.4", change: "+0.8%", icon: Activity, color: "text-blue-600", bg: "bg-blue-100" },
          { label: "Asia Tech Exposure", value: "22.5%", change: "High", icon: AlertCircle, color: "text-amber-600", bg: "bg-amber-100" },
          { label: "Risk Score", value: "6.2/10", change: "Moderate", icon: ShieldAlert, color: "text-indigo-600", bg: "bg-indigo-100" },
        ].map((metric, i) => (
          <div key={i} className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-medium text-slate-500">{metric.label}</p>
                <h3 className="text-2xl font-bold text-slate-900 mt-2">{metric.value}</h3>
              </div>
              <div className={`p-3 rounded-xl ${metric.bg}`}>
                <metric.icon className={`w-5 h-5 ${metric.color}`} />
              </div>
            </div>
            <p className={`text-sm mt-4 font-medium ${metric.change.startsWith('+') ? 'text-emerald-600' : 'text-slate-600'}`}>
              {metric.change} today
            </p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content Area */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Chart Section */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 h-[300px]">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Intraday Performance</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockChartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dy={10} />
                <YAxis domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dx={-10} />
                <Tooltip 
                  contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{r: 6, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2}} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* AI Prompt Section */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Ask FinAgent</h3>
            <form onSubmit={handleSubmit} className="relative">
              <div className="relative flex items-center">
                <textarea
                  rows={3}
                  className="w-full pl-4 pr-32 py-4 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all resize-none text-slate-700"
                  placeholder="e.g., What's our risk exposure in Asia tech stocks today?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                />
                <div className="absolute right-3 bottom-3 flex space-x-2">
                  <button type="button" className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors">
                    <Mic className="w-5 h-5" />
                  </button>
                  <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    <span>{loading ? "Thinking..." : "Send"}</span>
                    {!loading && <ArrowRight className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </form>

            {error && (
              <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 shrink-0" />
                <p className="text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Connected Accounts & APIs */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Connected Accounts & APIs</h3>
              <button
                onClick={() => setShowAddModal(true)}
                className="flex items-center space-x-1 text-sm font-medium text-blue-600 hover:text-blue-700 px-3 py-1.5 rounded-lg hover:bg-blue-50 transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Add Connection</span>
              </button>
            </div>

            {connections.length > 0 ? (
              <div className="space-y-3">
                {connections.map((conn) => {
                  const statusColor =
                    conn.status === "connected" ? "bg-emerald-500" :
                    conn.status === "stored" ? "bg-blue-500" :
                    conn.status === "auth_failed" ? "bg-red-500" :
                    conn.status === "rate_limited" ? "bg-amber-500" :
                    "bg-slate-400";
                  const statusLabel =
                    conn.status === "connected" ? "Connected" :
                    conn.status === "stored" ? "Stored (unverified)" :
                    conn.status === "auth_failed" ? "Auth Failed" :
                    conn.status === "rate_limited" ? "Rate Limited" :
                    conn.status === "unreachable" ? "Unreachable" :
                    conn.status;
                  return (
                    <div key={conn.id} className="p-4 border border-slate-200 rounded-xl flex items-center justify-between hover:border-slate-300 transition-colors group">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-slate-100 rounded-lg flex items-center justify-center">
                          <span className="font-bold text-slate-600 text-xs uppercase">{conn.provider.slice(0, 3)}</span>
                        </div>
                        <div>
                          <h4 className="font-medium text-slate-900 text-sm">{conn.name}</h4>
                          <p className="text-xs text-slate-400 font-mono">{conn.api_key_masked}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                          conn.status === "connected" ? "bg-emerald-100 text-emerald-700" :
                          conn.status === "auth_failed" ? "bg-red-100 text-red-700" :
                          "bg-slate-100 text-slate-600"
                        }`}>
                          {statusLabel}
                        </span>
                        <button
                          onClick={() => handleTestConnection(conn.id)}
                          className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                          title="Re-test"
                        >
                          {testingId === conn.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                        </button>
                        <button
                          onClick={() => handleDeleteConnection(conn.id)}
                          className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                        <div className={`w-2 h-2 rounded-full ${statusColor}`}></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-400">
                <p className="text-sm mb-2">No API connections configured yet.</p>
                <button onClick={() => setShowAddModal(true)} className="text-blue-600 text-sm font-medium hover:underline">
                  Add your first connection
                </button>
              </div>
            )}
          </div>

          {/* Add Connection Modal */}
          {showAddModal && (
            <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setShowAddModal(false)}>
              <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg p-6" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold text-slate-900">Add API Connection</h3>
                  <button onClick={() => setShowAddModal(false)} className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
                    <X className="w-5 h-5 text-slate-500" />
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">Provider</label>
                    <select
                      value={addForm.provider}
                      onChange={(e) => setAddForm({ ...addForm, provider: e.target.value, name: e.target.value === "openai" ? "OpenAI" : e.target.value === "alphavantage" ? "AlphaVantage" : e.target.value === "elevenlabs" ? "ElevenLabs" : addForm.name })}
                      className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700"
                    >
                      <option value="openai">OpenAI (LLM Engine)</option>
                      <option value="alphavantage">AlphaVantage (Market Data)</option>
                      <option value="elevenlabs">ElevenLabs (Voice / TTS)</option>
                      <option value="broker">Broker (Alpaca / IBKR)</option>
                      <option value="custom">Custom Provider</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">Connection Name</label>
                    <input
                      type="text"
                      value={addForm.name}
                      onChange={(e) => setAddForm({ ...addForm, name: e.target.value })}
                      placeholder="e.g. OpenAI Production"
                      className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">API Key</label>
                    <input
                      type="password"
                      value={addForm.api_key}
                      onChange={(e) => setAddForm({ ...addForm, api_key: e.target.value })}
                      placeholder="sk-..."
                      className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700 font-mono text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">Base URL <span className="text-slate-400">(optional)</span></label>
                    <input
                      type="text"
                      value={addForm.base_url}
                      onChange={(e) => setAddForm({ ...addForm, base_url: e.target.value })}
                      placeholder="https://api.openai.com"
                      className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700 font-mono text-sm"
                    />
                    <p className="text-xs text-slate-400 mt-1">Leave blank to use the default endpoint for this provider.</p>
                  </div>

                  {addError && (
                    <div className="p-3 bg-red-50 text-red-700 rounded-xl border border-red-100 text-sm flex items-center space-x-2">
                      <XCircle className="w-4 h-4 shrink-0" />
                      <span>{addError}</span>
                    </div>
                  )}

                  <div className="flex justify-end space-x-3 pt-2">
                    <button onClick={() => setShowAddModal(false)} className="px-5 py-2.5 border border-slate-200 rounded-xl font-medium text-slate-600 hover:bg-slate-50 transition-colors">
                      Cancel
                    </button>
                    <button
                      onClick={handleAddConnection}
                      disabled={addLoading || !addForm.name.trim() || !addForm.api_key.trim()}
                      className="flex items-center space-x-2 px-5 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {addLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                      <span>{addLoading ? "Connecting..." : "Add & Verify"}</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Results */}
        <div className="lg:col-span-1 space-y-6">
          {response ? (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 h-full animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center justify-between mb-4 pb-4 border-b border-slate-100">
                <h3 className="text-lg font-semibold text-slate-900 flex items-center">
                  <Bot className="w-5 h-5 text-blue-600 mr-2" />
                  Agent Response
                </h3>
                <span className="px-2.5 py-1 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full">
                  {(response.confidence * 100).toFixed(0)}% Conf
                </span>
              </div>
              
              <div className="prose prose-slate prose-sm text-slate-700 max-w-none">
                <p className="leading-relaxed whitespace-pre-wrap">{response.text_response}</p>
              </div>

              {response.audio_response && (
                <div className="mt-6 pt-4 border-t border-slate-100">
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Audio Brief</h4>
                  <audio controls className="w-full h-10 rounded-lg" src={`data:audio/mp3;base64,${response.audio_response}`} />
                </div>
              )}

              {response.metrics && response.metrics.total_latency && (
                <div className="mt-6 pt-4 border-t border-slate-100">
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Metrics</h4>
                  <div className="text-sm text-slate-600 flex justify-between">
                    <span>Processing Time:</span>
                    <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-slate-800">
                      {Number(response.metrics.total_latency).toFixed(2)}s
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-slate-100/50 p-6 rounded-2xl border border-slate-200/60 h-full flex flex-col items-center justify-center text-center text-slate-400 space-y-4 min-h-[400px]">
              <div className="w-16 h-16 bg-white rounded-full shadow-sm flex items-center justify-center text-slate-300">
                <MessageSquare className="w-8 h-8" />
              </div>
              <p className="text-sm max-w-[200px]">
                Submit a query to generate a multi-agent market brief and analysis.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
