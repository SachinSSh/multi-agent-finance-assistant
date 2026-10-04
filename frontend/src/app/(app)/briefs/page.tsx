"use client";

import { useState } from "react";
import { FileText, Clock, Tag, Plus, ChevronRight, X, ArrowRight } from "lucide-react";

const mockBriefs = [
  {
    id: 1,
    title: "Morning Brief — Oct 4, 2026",
    summary: "Asia tech exposure at 22% AUM, up from 18% yesterday. TSMC beat estimates by 4%, Samsung missed by 2%. Regional sentiment neutral with cautionary tilt due to rising yields.",
    confidence: 0.92,
    timestamp: "2026-10-04T09:00:00",
    tags: ["Asia Tech", "Earnings", "Risk"],
  },
  {
    id: 2,
    title: "Earnings Alert — NVDA Q3 Results",
    summary: "NVIDIA reported Q3 revenue of $35.1B, beating consensus of $33.2B. Data center revenue surged 112% YoY. Guidance for Q4 slightly above expectations.",
    confidence: 0.95,
    timestamp: "2026-10-03T16:30:00",
    tags: ["Earnings", "Semiconductors"],
  },
  {
    id: 3,
    title: "Risk Report — Portfolio Volatility Spike",
    summary: "Portfolio beta increased to 1.15 from 0.95 over the past week. Concentrated tech exposure contributing to elevated VaR. Consider rebalancing into defensive sectors.",
    confidence: 0.88,
    timestamp: "2026-10-03T14:00:00",
    tags: ["Risk", "Volatility", "Rebalancing"],
  },
  {
    id: 4,
    title: "Macro Update — Fed Minutes Analysis",
    summary: "Fed minutes indicate potential rate pause in November. Bond yields retreated 8bps. Dollar weakened against major currencies. Positive for growth and tech equities.",
    confidence: 0.85,
    timestamp: "2026-10-02T20:00:00",
    tags: ["Macro", "Fed", "Rates"],
  },
  {
    id: 5,
    title: "Sector Rotation Alert",
    summary: "Significant capital outflows from energy sector ($2.1B weekly). Inflows concentrated in healthcare ($1.4B) and technology ($3.2B). Momentum shifting toward quality growth.",
    confidence: 0.81,
    timestamp: "2026-10-02T10:00:00",
    tags: ["Sectors", "Flows", "Rotation"],
  },
  {
    id: 6,
    title: "Morning Brief — Oct 2, 2026",
    summary: "S&P 500 closed at 5,102 (+0.8%). Portfolio outperformed benchmark by 32bps. Top contributors: NVDA (+3.2%), AAPL (+1.1%). Detractors: JPM (-0.8%).",
    confidence: 0.93,
    timestamp: "2026-10-02T09:00:00",
    tags: ["Daily", "Performance"],
  },
];

function formatDate(ts: string) {
  return new Date(ts).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function BriefsPage() {
  const [showInput, setShowInput] = useState(false);
  const [newQuery, setNewQuery] = useState("");

  return (
    <main className="flex-1 p-8 overflow-y-auto">
      <header className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Market Briefs</h2>
          <p className="text-slate-500 mt-1">Your AI-generated market intelligence</p>
        </div>
        <button
          onClick={() => setShowInput(!showInput)}
          className="flex items-center space-x-2 px-5 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
        >
          {showInput ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          <span>{showInput ? "Cancel" : "Generate New Brief"}</span>
        </button>
      </header>

      {showInput && (
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 mb-8">
          <h3 className="text-lg font-semibold text-slate-900 mb-3">New Brief Query</h3>
          <div className="flex space-x-3">
            <input
              type="text"
              className="flex-1 px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-700"
              placeholder="e.g., Summarize today's earnings surprises in the semiconductor sector"
              value={newQuery}
              onChange={(e) => setNewQuery(e.target.value)}
            />
            <button className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors">
              <span>Generate</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {mockBriefs.map((brief) => (
          <div
            key={brief.id}
            className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md hover:border-slate-200 transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4 flex-1">
                <div className="p-3 bg-blue-50 rounded-xl shrink-0">
                  <FileText className="w-5 h-5 text-blue-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-lg font-semibold text-slate-900">{brief.title}</h3>
                    <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full shrink-0">
                      {(brief.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="text-slate-600 text-sm leading-relaxed mb-3">{brief.summary}</p>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1 text-xs text-slate-400">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{formatDate(brief.timestamp)}</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <Tag className="w-3.5 h-3.5 text-slate-400" />
                      {brief.tags.map((tag) => (
                        <span key={tag} className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs rounded-md">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              <ChevronRight className="w-5 h-5 text-slate-300 group-hover:text-blue-500 transition-colors shrink-0 mt-1" />
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
