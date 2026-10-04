"use client";

import { useState } from "react";
import { Settings, Key, Bot, Volume2, RefreshCw, Trash2, Shield, Eye, EyeOff } from "lucide-react";

export default function SettingsPage() {
  const [orchestratorUrl, setOrchestratorUrl] = useState("http://localhost:8000");
  const [openAiKey, setOpenAiKey] = useState("");
  const [alphaVantageKey, setAlphaVantageKey] = useState("");
  const [elevenLabsKey, setElevenLabsKey] = useState("");
  
  const [showOpenAi, setShowOpenAi] = useState(false);
  const [showAlpha, setShowAlpha] = useState(false);
  const [showEleven, setShowEleven] = useState(false);

  const [agents, setAgents] = useState({
    api: true,
    scraper: true,
    retriever: true,
    analysis: true,
    language: true,
    voice: false,
  });

  const [voiceOutput, setVoiceOutput] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState("5m");
  const [theme, setTheme] = useState("light");

  const [isSaving, setIsSaving] = useState(false);

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      alert("Settings saved successfully!");
    }, 800);
  };

  return (
    <main className="flex-1 p-8 overflow-y-auto">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Settings</h2>
          <p className="text-slate-500 mt-1">Configure your FinAgent workspace</p>
        </div>
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="px-6 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors disabled:opacity-70 flex items-center"
        >
          {isSaving ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            "Save Changes"
          )}
        </button>
      </header>

      <div className="max-w-4xl space-y-8 pb-12">
        {/* API Configuration */}
        <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2.5 bg-blue-50 text-blue-600 rounded-lg">
              <Key className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900">API Configuration</h3>
          </div>
          
          <div className="space-y-6 max-w-2xl">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Orchestrator URL</label>
              <input
                type="text"
                value={orchestratorUrl}
                onChange={(e) => setOrchestratorUrl(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700"
              />
              <p className="text-xs text-slate-500 mt-1.5">The local endpoint for the FastAPI swarm orchestrator.</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">OpenAI API Key</label>
              <div className="relative">
                <input
                  type={showOpenAi ? "text" : "password"}
                  value={openAiKey}
                  onChange={(e) => setOpenAiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full px-4 py-2.5 pr-12 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700 font-mono text-sm"
                />
                <button type="button" onClick={() => setShowOpenAi(!showOpenAi)} className="absolute right-3 top-3 text-slate-400 hover:text-slate-600">
                  {showOpenAi ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">AlphaVantage API Key</label>
              <div className="relative">
                <input
                  type={showAlpha ? "text" : "password"}
                  value={alphaVantageKey}
                  onChange={(e) => setAlphaVantageKey(e.target.value)}
                  className="w-full px-4 py-2.5 pr-12 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700 font-mono text-sm"
                />
                <button type="button" onClick={() => setShowAlpha(!showAlpha)} className="absolute right-3 top-3 text-slate-400 hover:text-slate-600">
                  {showAlpha ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">ElevenLabs API Key</label>
              <div className="relative">
                <input
                  type={showEleven ? "text" : "password"}
                  value={elevenLabsKey}
                  onChange={(e) => setElevenLabsKey(e.target.value)}
                  className="w-full px-4 py-2.5 pr-12 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700 font-mono text-sm"
                />
                <button type="button" onClick={() => setShowEleven(!showEleven)} className="absolute right-3 top-3 text-slate-400 hover:text-slate-600">
                  {showEleven ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Agent Configuration */}
        <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2.5 bg-emerald-50 text-emerald-600 rounded-lg">
              <Bot className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900">Agent Configuration</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(agents).map(([agent, isEnabled]) => (
              <div key={agent} className="flex items-center justify-between p-4 border border-slate-100 rounded-xl bg-slate-50">
                <div>
                  <h4 className="font-medium text-slate-900 capitalize">{agent} Agent</h4>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {isEnabled ? "Active in orchestrator swarm" : "Disabled"}
                  </p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    className="sr-only peer"
                    checked={isEnabled}
                    onChange={(e) => setAgents({ ...agents, [agent]: e.target.checked })}
                  />
                  <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
                </label>
              </div>
            ))}
          </div>
        </section>

        {/* Preferences */}
        <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2.5 bg-violet-50 text-violet-600 rounded-lg">
              <Settings className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900">Preferences</h3>
          </div>

          <div className="space-y-6 max-w-2xl">
            <div className="flex items-center justify-between p-4 border border-slate-100 rounded-xl bg-slate-50">
              <div className="flex items-center space-x-3">
                <Volume2 className="w-5 h-5 text-slate-400" />
                <div>
                  <h4 className="font-medium text-slate-900">Voice Output</h4>
                  <p className="text-xs text-slate-500">Auto-play audio briefs when generated</p>
                </div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={voiceOutput}
                  onChange={(e) => setVoiceOutput(e.target.checked)}
                />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Auto-refresh Interval</label>
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(e.target.value)}
                  className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700"
                >
                  <option value="30s">30 seconds</option>
                  <option value="1m">1 minute</option>
                  <option value="5m">5 minutes</option>
                  <option value="10m">10 minutes</option>
                  <option value="never">Never</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Theme</label>
                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                  className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="system">System Default</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        {/* Danger Zone */}
        <section className="bg-white p-8 rounded-2xl border border-red-200 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-red-500"></div>
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2.5 bg-red-50 text-red-600 rounded-lg">
              <Shield className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900">Danger Zone</h3>
          </div>
          
          <div className="flex space-x-4">
            <button className="px-5 py-2.5 border border-red-200 text-red-600 hover:bg-red-50 rounded-xl font-medium transition-colors">
              Reset All Settings
            </button>
            <button className="flex items-center space-x-2 px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-xl font-medium transition-colors">
              <Trash2 className="w-4 h-4" />
              <span>Clear Cache</span>
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}
