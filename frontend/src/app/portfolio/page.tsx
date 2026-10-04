"use client";

import { TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight, PieChart } from "lucide-react";

const holdings = [
  { symbol: "AAPL", name: "Apple Inc.", shares: 450, price: 178.52, value: 80334, change: 1.23, allocation: 6.5 },
  { symbol: "MSFT", name: "Microsoft Corp.", shares: 320, price: 378.91, value: 121251, change: 0.85, allocation: 9.8 },
  { symbol: "GOOGL", name: "Alphabet Inc.", shares: 280, price: 141.80, value: 39704, change: -0.42, allocation: 3.2 },
  { symbol: "TSMC", name: "Taiwan Semi", shares: 600, price: 108.25, value: 64950, change: 2.15, allocation: 5.2 },
  { symbol: "AMZN", name: "Amazon.com Inc.", shares: 380, price: 185.60, value: 70528, change: 1.67, allocation: 5.7 },
  { symbol: "NVDA", name: "NVIDIA Corp.", shares: 520, price: 495.22, value: 257514, change: 3.21, allocation: 20.8 },
  { symbol: "META", name: "Meta Platforms", shares: 290, price: 512.40, value: 148596, change: -0.95, allocation: 12.0 },
  { symbol: "JPM", name: "JPMorgan Chase", shares: 410, price: 198.35, value: 81324, change: -0.78, allocation: 6.6 },
  { symbol: "V", name: "Visa Inc.", shares: 340, price: 282.10, value: 95914, change: 0.54, allocation: 7.7 },
  { symbol: "UNH", name: "UnitedHealth", shares: 150, price: 548.90, value: 82335, change: 1.02, allocation: 6.6 },
];

const sectors = [
  { name: "Technology", allocation: 45.2, color: "bg-blue-500" },
  { name: "Financials", allocation: 15.3, color: "bg-emerald-500" },
  { name: "Healthcare", allocation: 12.1, color: "bg-violet-500" },
  { name: "Consumer Disc.", allocation: 10.8, color: "bg-amber-500" },
  { name: "Industrials", allocation: 8.4, color: "bg-rose-500" },
  { name: "Cash & Equiv.", allocation: 8.2, color: "bg-slate-400" },
];

const totalValue = 1240000;
const dayChange = 12450;
const dayChangePct = 1.01;
const totalReturn = 18.3;

export default function PortfolioPage() {
  return (
    <main className="flex-1 p-8 overflow-y-auto">
      <header className="mb-8">
        <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Portfolio Overview</h2>
        <p className="text-slate-500 mt-1">Track your holdings and allocation</p>
      </header>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <p className="text-sm font-medium text-slate-500">Total Value</p>
          <h3 className="text-2xl font-bold text-slate-900 mt-2">${(totalValue / 1000000).toFixed(2)}M</h3>
        </div>
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <p className="text-sm font-medium text-slate-500">Day Change</p>
          <h3 className="text-2xl font-bold text-emerald-600 mt-2 flex items-center">
            <ArrowUpRight className="w-5 h-5 mr-1" />
            ${dayChange.toLocaleString()} ({dayChangePct}%)
          </h3>
        </div>
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <p className="text-sm font-medium text-slate-500">Total Return</p>
          <h3 className="text-2xl font-bold text-emerald-600 mt-2 flex items-center">
            <TrendingUp className="w-5 h-5 mr-1" />
            +{totalReturn}%
          </h3>
        </div>
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <p className="text-sm font-medium text-slate-500">Holdings</p>
          <h3 className="text-2xl font-bold text-slate-900 mt-2">{holdings.length} positions</h3>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Holdings Table */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
          <div className="p-6 border-b border-slate-100">
            <h3 className="text-lg font-semibold text-slate-900">Holdings</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 text-slate-500 text-left">
                  <th className="px-6 py-3 font-medium">Symbol</th>
                  <th className="px-6 py-3 font-medium">Name</th>
                  <th className="px-6 py-3 font-medium text-right">Shares</th>
                  <th className="px-6 py-3 font-medium text-right">Price</th>
                  <th className="px-6 py-3 font-medium text-right">Value</th>
                  <th className="px-6 py-3 font-medium text-right">Change</th>
                  <th className="px-6 py-3 font-medium text-right">Alloc.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {holdings.map((h) => (
                  <tr key={h.symbol} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 font-bold text-slate-900">{h.symbol}</td>
                    <td className="px-6 py-4 text-slate-600">{h.name}</td>
                    <td className="px-6 py-4 text-right text-slate-700">{h.shares}</td>
                    <td className="px-6 py-4 text-right text-slate-700">${h.price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right font-medium text-slate-900">${h.value.toLocaleString()}</td>
                    <td className={`px-6 py-4 text-right font-medium flex items-center justify-end ${h.change >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                      {h.change >= 0 ? <ArrowUpRight className="w-3.5 h-3.5 mr-0.5" /> : <ArrowDownRight className="w-3.5 h-3.5 mr-0.5" />}
                      {h.change >= 0 ? "+" : ""}{h.change}%
                    </td>
                    <td className="px-6 py-4 text-right text-slate-500">{h.allocation}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Sector Allocation */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <div className="flex items-center space-x-2 mb-6">
            <PieChart className="w-5 h-5 text-slate-600" />
            <h3 className="text-lg font-semibold text-slate-900">Sector Allocation</h3>
          </div>
          <div className="space-y-4">
            {sectors.map((s) => (
              <div key={s.name}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-700 font-medium">{s.name}</span>
                  <span className="text-slate-500">{s.allocation}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2.5">
                  <div className={`${s.color} h-2.5 rounded-full transition-all`} style={{ width: `${s.allocation}%` }}></div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-6 pt-4 border-t border-slate-100">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-500">Diversification Score</span>
              <span className="font-semibold text-slate-900">7.2 / 10</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
