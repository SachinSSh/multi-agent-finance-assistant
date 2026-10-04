"use client";

import { BarChart3, Activity, Shield, Gauge } from "lucide-react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const portfolioPerformance = [
  { month: "Jan", value: 100000 }, { month: "Feb", value: 103500 }, { month: "Mar", value: 101200 },
  { month: "Apr", value: 106800 }, { month: "May", value: 109400 }, { month: "Jun", value: 107100 },
  { month: "Jul", value: 112300 }, { month: "Aug", value: 115800 }, { month: "Sep", value: 118200 },
  { month: "Oct", value: 116500 }, { month: "Nov", value: 120900 }, { month: "Dec", value: 124000 },
];

const monthlyReturns = [
  { month: "Jan", return: 2.1 }, { month: "Feb", return: 3.5 }, { month: "Mar", return: -2.2 },
  { month: "Apr", return: 5.5 }, { month: "May", return: 2.4 }, { month: "Jun", return: -2.1 },
  { month: "Jul", return: 4.9 }, { month: "Aug", return: 3.1 }, { month: "Sep", return: 2.1 },
  { month: "Oct", return: -1.4 }, { month: "Nov", return: 3.8 }, { month: "Dec", return: 2.6 },
];

const riskMetrics = [
  { label: "Sharpe Ratio", value: "1.42", icon: Gauge, color: "text-blue-600", bg: "bg-blue-50" },
  { label: "Beta", value: "0.95", icon: Activity, color: "text-emerald-600", bg: "bg-emerald-50" },
  { label: "Max Drawdown", value: "-8.2%", icon: Shield, color: "text-red-600", bg: "bg-red-50" },
  { label: "Volatility", value: "14.3%", icon: BarChart3, color: "text-amber-600", bg: "bg-amber-50" },
  { label: "Alpha", value: "2.1%", icon: Activity, color: "text-violet-600", bg: "bg-violet-50" },
  { label: "Sortino Ratio", value: "1.87", icon: Gauge, color: "text-indigo-600", bg: "bg-indigo-50" },
];

const agentPerformance = [
  { name: "API Agent", latency: "142ms", success: "99.2%", queries: "12,450", status: "Healthy" },
  { name: "Scraper Agent", latency: "890ms", success: "94.1%", queries: "3,210", status: "Healthy" },
  { name: "Retriever Agent", latency: "234ms", success: "98.7%", queries: "8,930", status: "Healthy" },
  { name: "Analysis Agent", latency: "567ms", success: "97.3%", queries: "6,120", status: "Healthy" },
  { name: "Language Agent", latency: "1,230ms", success: "96.5%", queries: "5,890", status: "Degraded" },
  { name: "Voice Agent", latency: "780ms", success: "91.8%", queries: "1,450", status: "Offline" },
];

function getStatusColor(status: string) {
  switch (status) {
    case "Healthy": return "bg-emerald-100 text-emerald-700";
    case "Degraded": return "bg-amber-100 text-amber-700";
    case "Offline": return "bg-red-100 text-red-700";
    default: return "bg-slate-100 text-slate-700";
  }
}

export default function AnalyticsPage() {
  return (
    <main className="flex-1 p-8 overflow-y-auto">
      <header className="mb-8">
        <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Analytics</h2>
        <p className="text-slate-500 mt-1">Deep-dive into performance and risk metrics</p>
      </header>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Portfolio Performance</h3>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={portfolioPerformance}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip formatter={(value: number) => [`$${value.toLocaleString()}`, "Value"]} contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }} />
                <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2.5} fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Monthly Returns (%)</h3>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlyReturns}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(value: number) => [`${value}%`, "Return"]} contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }} />
                <Bar dataKey="return" radius={[6, 6, 0, 0]} fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">Risk Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {riskMetrics.map((m) => (
            <div key={m.label} className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 text-center">
              <div className={`w-10 h-10 ${m.bg} rounded-xl flex items-center justify-center mx-auto mb-3`}>
                <m.icon className={`w-5 h-5 ${m.color}`} />
              </div>
              <p className="text-2xl font-bold text-slate-900">{m.value}</p>
              <p className="text-xs text-slate-500 mt-1">{m.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Agent Performance Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h3 className="text-lg font-semibold text-slate-900">Agent Performance</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-slate-500 text-left">
                <th className="px-6 py-3 font-medium">Agent</th>
                <th className="px-6 py-3 font-medium text-right">Avg Latency</th>
                <th className="px-6 py-3 font-medium text-right">Success Rate</th>
                <th className="px-6 py-3 font-medium text-right">Total Queries</th>
                <th className="px-6 py-3 font-medium text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {agentPerformance.map((a) => (
                <tr key={a.name} className="hover:bg-slate-50 transition-colors">
                  <td className="px-6 py-4 font-medium text-slate-900">{a.name}</td>
                  <td className="px-6 py-4 text-right font-mono text-slate-600">{a.latency}</td>
                  <td className="px-6 py-4 text-right text-slate-700">{a.success}</td>
                  <td className="px-6 py-4 text-right text-slate-600">{a.queries}</td>
                  <td className="px-6 py-4 text-right">
                    <span className={`px-2.5 py-1 text-xs font-semibold rounded-full ${getStatusColor(a.status)}`}>
                      {a.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
