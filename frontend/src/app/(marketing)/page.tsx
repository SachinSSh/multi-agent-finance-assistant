import Link from "next/link";
import { ArrowRight, Bot, Shield, Zap, LineChart, Cpu, BarChart3, Code, Terminal, Database, Mic, Headphones, Network, Activity, Lock, Workflow, Server, Key, EyeOff, Plug, HelpCircle, ChevronDown } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="w-full min-h-screen bg-slate-950 text-slate-50 flex flex-col font-sans selection:bg-blue-500/30 scroll-smooth">
      <header className="py-6 px-8 flex items-center justify-between border-b border-white/5 bg-slate-950/90 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-600 p-2 rounded-xl shadow-[0_0_15px_-3px_rgba(37,99,235,0.5)]">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">FinAgent OS</span>
        </div>
        <nav className="hidden lg:flex space-x-8 text-sm font-medium text-slate-300">
          <Link href="#agents" className="hover:text-white transition-colors">The Swarm</Link>
          <Link href="#knowledge" className="hover:text-white transition-colors">Knowledge Base</Link>
          <Link href="#voice" className="hover:text-white transition-colors">Voice AI</Link>
          <Link href="#security" className="hover:text-white transition-colors">Security</Link>
          <Link href="#developer" className="hover:text-white transition-colors">Developers</Link>
          <Link href="#faq" className="hover:text-white transition-colors">FAQ</Link>
        </nav>
        <div className="flex items-center space-x-4">
          <a href="https://github.com" target="_blank" rel="noreferrer" className="hidden md:flex text-slate-400 hover:text-white transition-colors items-center gap-2">
            <Code className="w-5 h-5" />
          </a>
          <Link 
            href="/dashboard" 
            className="bg-white text-slate-950 px-5 py-2.5 rounded-full font-semibold text-sm hover:bg-slate-200 transition-colors shadow-lg"
          >
            Open Dashboard
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex flex-col items-center justify-center text-center px-4 pt-32 pb-24 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-slate-950 to-slate-950 relative overflow-hidden">
        <div className="absolute top-1/4 left-0 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px] pointer-events-none"></div>
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-emerald-600/5 rounded-full blur-[150px] pointer-events-none"></div>

        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium mb-8 relative z-10">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
          </span>
          <span>v2.0 Multi-Agent Orchestrator Live</span>
        </div>
        
        <h1 className="text-5xl md:text-7xl lg:text-8xl font-extrabold tracking-tight max-w-5xl leading-[1.05] mb-8 relative z-10">
          The autonomous <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-emerald-400 to-teal-300">multi-agent</span> finance assistant.
        </h1>
        
        <p className="text-lg md:text-xl text-slate-400 max-w-2xl mb-12 leading-relaxed relative z-10">
          Your locally-hosted AI swarm. Ask complex market questions and let specialized agents fetch data, analyze risk, and synthesize real-time insights for your portfolio.
        </p>
        
        <div className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-4 mb-20 relative z-10">
          <Link 
            href="/dashboard" 
            className="flex items-center space-x-2 bg-blue-600 text-white px-8 py-4 rounded-full font-semibold text-lg hover:bg-blue-700 transition-colors shadow-[0_0_40px_-10px_rgba(37,99,235,0.5)]"
          >
            <span>Launch Platform</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
          <a 
            href="https://github.com/your-username/finance-assistant"
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-2 px-8 py-4 rounded-full font-semibold text-lg border border-slate-700 hover:bg-slate-800 transition-colors bg-slate-900/50 backdrop-blur-sm"
          >
            <Code className="w-5 h-5" />
            <span>View Source</span>
          </a>
        </div>

        {/* Hero Dashboard Preview */}
        <div className="w-full max-w-6xl mx-auto rounded-xl border border-white/10 bg-slate-900/50 p-2 md:p-4 backdrop-blur-sm shadow-2xl relative z-10">
          <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent z-10 rounded-xl pointer-events-none"></div>
          <div className="aspect-[16/9] lg:aspect-[21/9] rounded-lg bg-slate-950 border border-slate-800 overflow-hidden flex flex-col relative">
            <div className="h-8 border-b border-slate-800 bg-slate-900 flex items-center px-4 gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
              <div className="ml-4 text-xs font-mono text-slate-500 flex-1 text-center">orchestrator/main.py — FastAPI</div>
            </div>
            <div className="flex-1 p-6 text-left font-mono text-sm text-green-400 overflow-hidden relative">
              <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5"></div>
              <p className="mb-2 text-slate-400"># FinAgent OS Backend Console</p>
              <p className="mb-1">{`> Starting FastAPI orchestrator on port 8000...`}</p>
              <p className="mb-1 text-blue-400">{`> [System] Swarm initialized with 6 agents.`}</p>
              <p className="mb-1">{`> Received user query: "Analyze tech portfolio risk profile"`}</p>
              <p className="mb-1 text-amber-400">{`> [Router] Delegating to: api_agent, analysis_agent, scraper_agent`}</p>
              <p className="mb-1 text-emerald-400">{`> [api_agent] Fetched live quotes for 24 holdings (120ms)`}</p>
              <p className="mb-1 text-emerald-400">{`> [analysis_agent] Calculated portfolio Beta: 1.15, VaR: -3.2% (85ms)`}</p>
              <p className="mb-1 text-emerald-400">{`> [scraper_agent] Scraped Q3 earnings data for MSFT, AAPL (210ms)`}</p>
              <p className="mb-1 text-emerald-400">{`> [language_agent] Synthesized risk report (412ms)`}</p>
              <p className="mb-1">{`> Response delivered to frontend.`}</p>
              <span className="animate-pulse">_</span>
            </div>
          </div>
        </div>
      </main>

      {/* Agents Section */}
      <section id="agents" className="py-32 px-8 border-t border-white/5 bg-slate-950 relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold mb-6">The Six-Agent Swarm</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Instead of relying on a single slow LLM prompt, FinAgent routes your query to a local swarm of specialized intelligences working in parallel to deliver high-confidence insights.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { icon: LineChart, title: "API Agent", desc: "Fetches real-time market data, historical pricing, and tracks portfolio exposure securely." },
              { icon: Shield, title: "Analysis Agent", desc: "Calculates volatility, beta, and exposure risk metrics instantly across your portfolio." },
              { icon: Zap, title: "Retriever Agent", desc: "Performs RAG-based vector search over financial documents to retrieve essential context." },
              { icon: Cpu, title: "Language Agent", desc: "Synthesizes the raw JSON outputs from the swarm into a coherent, high-confidence narrative." },
              { icon: BarChart3, title: "Scraper Agent", desc: "Monitors and scrapes unstructured financial web data, such as real-time earnings reports and news." },
              { icon: Bot, title: "Voice Agent", desc: "Provides high-quality Speech-to-Text and Text-to-Speech for a seamless conversational interface." }
            ].map((f, i) => (
              <div key={i} className="p-8 rounded-3xl bg-slate-900/40 border border-slate-800 hover:border-blue-500/50 hover:bg-slate-900 transition-all group relative overflow-hidden shadow-lg">
                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                  <f.icon className="w-32 h-32" />
                </div>
                <f.icon className="w-10 h-10 text-blue-500 mb-6 group-hover:scale-110 group-hover:text-blue-400 transition-all" />
                <h3 className="text-xl font-semibold mb-3">{f.title}</h3>
                <p className="text-slate-400 leading-relaxed relative z-10">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Integration Ecosystem Section */}
      <section className="py-24 px-8 bg-slate-900 border-t border-white/5 relative">
        <div className="max-w-6xl mx-auto text-center">
          <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-12">Seamlessly integrates with</h3>
          <div className="flex flex-wrap justify-center gap-8 md:gap-16 opacity-70 grayscale hover:grayscale-0 transition-all duration-500">
            {[
              "OpenAI API", "AlphaVantage", "SEC EDGAR", "ElevenLabs", "FAISS", "Next.js", "FastAPI", "Pandas"
            ].map((tech, i) => (
              <div key={i} className="flex items-center space-x-2">
                <Plug className="w-5 h-5 text-slate-400" />
                <span className="text-xl font-semibold text-slate-300">{tech}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Knowledge Base Section */}
      <section id="knowledge" className="py-32 px-8 bg-slate-950 border-t border-white/5 relative">
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-96 h-96 bg-emerald-500/10 blur-[150px] rounded-full pointer-events-none"></div>
        <div className="max-w-6xl mx-auto flex flex-col lg:flex-row items-center gap-16 relative z-10">
          <div className="lg:w-1/2 order-2 lg:order-1">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-4 pt-12">
                <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl hover:border-emerald-500/50 transition-colors">
                  <Database className="w-8 h-8 text-emerald-400 mb-4" />
                  <h4 className="font-medium text-white mb-2">SEC Filings</h4>
                  <p className="text-sm text-slate-400">10-K and 10-Q documents indexed for semantic search.</p>
                </div>
                <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl hover:border-blue-500/50 transition-colors">
                  <Terminal className="w-8 h-8 text-blue-400 mb-4" />
                  <h4 className="font-medium text-white mb-2">Earnings Transcripts</h4>
                  <p className="text-sm text-slate-400">Historical management commentary analysis.</p>
                </div>
              </div>
              <div className="space-y-4">
                <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl hover:border-purple-500/50 transition-colors">
                  <Network className="w-8 h-8 text-purple-400 mb-4" />
                  <h4 className="font-medium text-white mb-2">Vector Search</h4>
                  <p className="text-sm text-slate-400">FAISS-powered local embeddings for instant RAG.</p>
                </div>
                <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl hover:border-amber-500/50 transition-colors">
                  <Lock className="w-8 h-8 text-amber-400 mb-4" />
                  <h4 className="font-medium text-white mb-2">Local Storage</h4>
                  <p className="text-sm text-slate-400">No external database. Everything stays on your SSD.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="lg:w-1/2 order-1 lg:order-2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium mb-6">
              <Database className="w-4 h-4" />
              <span>Deep Context Retrieval</span>
            </div>
            <h2 className="text-4xl font-bold mb-6">Built-in Knowledge RAG</h2>
            <p className="text-xl text-slate-400 mb-8 leading-relaxed">
              The `retriever_agent` doesn't just guess. It performs high-speed vector searches over your local corpus of financial documents, injecting ground-truth context directly into the Language Agent's synthesis prompt.
            </p>
            <ul className="space-y-4">
              {[
                "Eliminates LLM hallucinations for specific financial figures",
                "Instant cross-referencing against quarterly earnings",
                "Entirely local processing for strict privacy"
              ].map((item, i) => (
                <li key={i} className="flex items-center space-x-3 text-slate-300">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* Security & Privacy Section */}
      <section id="security" className="py-32 px-8 bg-slate-900 border-t border-white/5 relative">
        <div className="max-w-6xl mx-auto flex flex-col lg:flex-row items-center gap-16">
          <div className="lg:w-1/2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 text-sm font-medium mb-6">
              <EyeOff className="w-4 h-4" />
              <span>Zero Telemetry</span>
            </div>
            <h2 className="text-4xl font-bold mb-6">Total Data Sovereignty</h2>
            <p className="text-xl text-slate-400 mb-8 leading-relaxed">
              Your financial life is private. Commercial AI tools train on your prompts and store your net worth on their servers. FinAgent OS is designed to be completely air-gappable (aside from explicit API data pulls).
            </p>
            <div className="space-y-6">
              <div className="flex gap-4">
                <Key className="w-6 h-6 text-red-400 shrink-0" />
                <div>
                  <h4 className="font-semibold text-white">Local API Keys</h4>
                  <p className="text-slate-400 text-sm mt-1">Keys are stored only in your local `.env`. They are never sent to a central server.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <Server className="w-6 h-6 text-red-400 shrink-0" />
                <div>
                  <h4 className="font-semibold text-white">Self-Hosted Compute</h4>
                  <p className="text-slate-400 text-sm mt-1">You own the infrastructure. The FastAPI orchestrator and Next.js frontend run purely on `localhost:8000` and `localhost:3000`.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="lg:w-1/2 w-full">
            <div className="p-8 rounded-3xl bg-slate-950 border border-slate-800 shadow-2xl relative overflow-hidden">
              <div className="absolute -right-20 -top-20 w-64 h-64 bg-red-500/10 blur-[100px] rounded-full pointer-events-none"></div>
              <div className="flex items-center justify-center py-12">
                <div className="relative">
                  <Shield className="w-32 h-32 text-slate-800" />
                  <Lock className="w-12 h-12 text-red-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
              </div>
              <div className="text-center pb-4 relative z-10">
                <p className="text-emerald-400 font-mono text-sm border border-emerald-900/50 bg-emerald-900/10 py-2 px-4 rounded-lg inline-block">
                  Status: 100% Local / Secure
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Voice capabilities */}
      <section id="voice" className="py-32 px-8 bg-slate-950 border-t border-white/5 relative overflow-hidden">
        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-96 h-96 bg-blue-600/10 blur-[150px] rounded-full pointer-events-none"></div>
        <div className="max-w-6xl mx-auto flex flex-col lg:flex-row items-center gap-16 relative z-10">
          <div className="lg:w-1/2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium mb-6">
              <Mic className="w-4 h-4" />
              <span>Conversational Interface</span>
            </div>
            <h2 className="text-4xl font-bold mb-6">Talk directly to your portfolio.</h2>
            <p className="text-xl text-slate-400 mb-8 leading-relaxed">
              Using the dedicated `voice_agent`, FinAgent OS supports native Speech-to-Text and Text-to-Speech orchestration. Ask complex questions out loud and receive a synthesized audio briefing moments later.
            </p>
            <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl mb-8 shadow-lg">
              <p className="text-sm text-slate-500 mb-2 font-mono">Example Voice Query</p>
              <p className="text-lg font-medium italic text-slate-300">"Hey FinAgent, how did my tech allocation perform today, and did any of them report earnings?"</p>
            </div>
          </div>
          <div className="lg:w-1/2 w-full">
            <div className="relative p-8 rounded-3xl bg-slate-900 border border-slate-800 shadow-2xl">
              <div className="absolute inset-0 bg-gradient-to-tr from-blue-900/10 to-transparent rounded-3xl"></div>
              
              <div className="flex items-center justify-between mb-8 relative z-10">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center animate-pulse shadow-[0_0_20px_-5px_rgba(37,99,235,0.7)]">
                    <Mic className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-white">Processing Audio...</h4>
                    <p className="text-xs text-blue-400 font-mono mt-1">voice_agent / STT Engine</p>
                  </div>
                </div>
                <div className="flex space-x-1">
                  {[15, 25, 12, 30, 20, 18, 35].map((height, i) => (
                    <div key={i} className={`w-1.5 bg-blue-500 rounded-full animate-pulse`} style={{ height: `${height}px`, animationDelay: `${i * 100}ms` }}></div>
                  ))}
                </div>
              </div>

              <div className="space-y-4 relative z-10">
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 w-3/4 rounded-full"></div>
                </div>
                <div className="flex justify-between text-xs text-slate-500 font-mono">
                  <span>Routing to Language Agent</span>
                  <span className="text-emerald-400">75%</span>
                </div>
                
                <div className="mt-8 pt-6 border-t border-slate-800 flex items-center justify-between">
                  <div className="flex items-center space-x-3 text-sm">
                    <Headphones className="w-4 h-4 text-slate-400" />
                    <span className="text-slate-400">Generating audio brief response</span>
                  </div>
                  <span className="text-xs font-mono text-slate-500">TTS Engine</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Data Flow Section */}
      <section id="architecture" className="py-32 px-8 bg-slate-900 border-t border-white/5 relative">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-sm font-medium mb-6">
              <Workflow className="w-4 h-4" />
              <span>Orchestration Pipeline</span>
            </div>
            <h2 className="text-4xl font-bold mb-6">Sub-second Routing</h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Behind the scenes, the FastAPI orchestrator breaks down your single query into parallel sub-tasks, drastically reducing latency.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-16 relative">
            {/* Connecting Line background */}
            <div className="hidden lg:block absolute top-1/2 left-0 w-full h-[2px] bg-slate-800 -translate-y-1/2 z-0"></div>
            
            {[
              { step: "1. Intake", title: "Next.js Frontend", desc: "User submits query via text or audio in the dashboard UI." },
              { step: "2. Routing", title: "FastAPI Orchestrator", desc: "Intent is classified and broken down into parallel sub-tasks." },
              { step: "3. Execution", title: "Swarm Processing", desc: "Agents fetch data, scrape news, and calculate risk metrics concurrently." },
              { step: "4. Synthesis", title: "Language Agent", desc: "Raw JSON from the swarm is compiled into the final Markdown response." }
            ].map((s, i) => (
              <div key={i} className="bg-slate-950 border border-slate-800 p-6 rounded-2xl relative z-10 flex flex-col items-center text-center hover:border-blue-500/50 transition-colors">
                <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center text-slate-300 font-bold mb-6 border-4 border-slate-900">
                  {i + 1}
                </div>
                <h4 className="text-sm font-bold text-blue-500 uppercase tracking-wider mb-2">{s.step}</h4>
                <h3 className="text-lg font-semibold text-white mb-3">{s.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>

          <div className="bg-slate-950 p-8 rounded-3xl border border-slate-800 flex flex-col md:flex-row items-center justify-between shadow-2xl">
            <div className="mb-6 md:mb-0">
              <h4 className="text-xl font-bold text-white mb-2">Performance Monitoring</h4>
              <p className="text-slate-400">Every agent's latency and confidence score is tracked in real-time.</p>
            </div>
            <div className="flex gap-4">
              <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 text-center w-32">
                <Activity className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">412ms</p>
                <p className="text-xs text-slate-500 uppercase tracking-wider mt-1">Latency</p>
              </div>
              <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 text-center w-32">
                <Shield className="w-6 h-6 text-blue-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">99.2%</p>
                <p className="text-xs text-slate-500 uppercase tracking-wider mt-1">Reliable</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Developer API Section */}
      <section id="developer" className="py-32 px-8 bg-slate-950 border-t border-white/5 relative">
        <div className="max-w-6xl mx-auto flex flex-col lg:flex-row items-center gap-16">
          <div className="lg:w-1/2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm font-medium mb-6">
              <Terminal className="w-4 h-4" />
              <span>Extensible Architecture</span>
            </div>
            <h2 className="text-4xl font-bold mb-6">Built for Hackers</h2>
            <p className="text-xl text-slate-400 mb-8 leading-relaxed">
              Want to add a crypto trading agent? A sentiment analysis bot? Just add a new endpoint to the FastAPI orchestrator and map it in the router. It takes 10 lines of Python.
            </p>
            <a 
              href="https://github.com/your-username/finance-assistant"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center space-x-2 text-blue-400 hover:text-blue-300 font-semibold transition-colors"
            >
              <span>Read the Developer Docs</span>
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
          <div className="lg:w-1/2 w-full">
            <div className="rounded-xl overflow-hidden bg-slate-900 border border-slate-800 shadow-2xl">
              <div className="h-10 bg-slate-950 border-b border-slate-800 flex items-center px-4">
                <div className="text-xs font-mono text-slate-500">orchestrator/main.py (Adding an Agent)</div>
              </div>
              <div className="p-6 overflow-x-auto">
                <pre className="text-sm font-mono text-slate-300 leading-relaxed">
                  <code>
<span className="text-blue-400"># 1. Register new agent URL</span><br/>
AGENT_SERVICES[<span className="text-emerald-400">"crypto_agent"</span>] = <span className="text-emerald-400">"http://localhost:8007"</span><br/>
<br/>
<span className="text-blue-400"># 2. Add to router logic</span><br/>
<span className="text-purple-400">if</span> <span className="text-amber-300">any</span>(term <span className="text-purple-400">in</span> query_lower <span className="text-purple-400">for</span> term <span className="text-purple-400">in</span> [<span className="text-emerald-400">"btc"</span>, <span className="text-emerald-400">"crypto"</span>]):<br/>
&nbsp;&nbsp;&nbsp;&nbsp;tasks[<span className="text-emerald-400">"crypto_agent"</span>] = &#123;<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-emerald-400">"endpoint"</span>: <span className="text-emerald-400">"/analyze-crypto"</span>,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-emerald-400">"params"</span>: &#123;<span className="text-emerald-400">"query"</span>: query&#125;<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&#125;
                  </code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-32 px-8 bg-slate-900 border-t border-white/5">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Frequently Asked Questions</h2>
            <p className="text-slate-400 text-lg">Everything you need to know about running FinAgent OS.</p>
          </div>
          <div className="space-y-4">
            {[
              { q: "Is this completely free?", a: "The software itself is 100% open-source and free. However, you will need to supply your own API keys for OpenAI (or local LLM provider), AlphaVantage, and ElevenLabs (for voice)." },
              { q: "Can I run this on a Raspberry Pi?", a: "The FastAPI backend and Next.js frontend are very lightweight. However, if you are running FAISS vector embeddings or a local LLM, you will need adequate RAM (16GB+ recommended)." },
              { q: "Do I have to use the Voice features?", a: "No. The Voice AI module is completely optional. If you don't provide an ElevenLabs key, the app defaults to standard text-based markdown output." },
              { q: "How do I add a new portfolio asset?", a: "Currently, you can update the mock data in the Next.js `Portfolio` page, or wire up a real database connection in the `api_agent` python script." }
            ].map((faq, i) => (
              <div key={i} className="bg-slate-950 border border-slate-800 rounded-2xl p-6 hover:border-slate-700 transition-colors">
                <h4 className="text-lg font-bold text-white mb-3 flex justify-between items-center cursor-pointer">
                  {faq.q}
                  <HelpCircle className="w-5 h-5 text-slate-600" />
                </h4>
                <p className="text-slate-400 leading-relaxed">{faq.a}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 px-8 bg-blue-900 relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
        <div className="absolute top-0 right-0 w-full h-full bg-gradient-to-l from-slate-950 to-transparent opacity-80"></div>
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h2 className="text-5xl md:text-6xl font-extrabold text-white mb-8 tracking-tight leading-tight">Ready to orchestrate your portfolio?</h2>
          <p className="text-xl text-blue-200 mb-12 max-w-2xl mx-auto leading-relaxed">Start chatting with your local agents to generate deep market insights instantly without leaving your local environment.</p>
          <div className="flex flex-col sm:flex-row justify-center gap-6">
            <Link 
              href="/dashboard" 
              className="inline-flex items-center justify-center space-x-2 bg-white text-blue-950 px-10 py-5 rounded-full font-bold text-lg hover:bg-slate-100 transition-transform hover:scale-105 shadow-[0_0_50px_-12px_rgba(255,255,255,0.4)]"
            >
              <span>Open Dashboard</span>
              <ArrowRight className="w-6 h-6" />
            </Link>
            <a 
              href="https://github.com"
              className="inline-flex items-center justify-center space-x-2 bg-blue-950 text-white px-10 py-5 rounded-full font-bold text-lg hover:bg-slate-900 border border-blue-800 transition-colors"
            >
              <Code className="w-6 h-6" />
              <span>Clone Repository</span>
            </a>
          </div>
        </div>
      </section>

      <footer className="py-16 px-8 bg-slate-950 border-t border-white/5 text-center flex flex-col md:flex-row justify-between items-center max-w-7xl mx-auto w-full text-slate-500 gap-6">
        <div className="flex items-center space-x-3">
          <Bot className="w-6 h-6 text-blue-500" />
          <span className="font-bold text-slate-300 text-lg">FinAgent OS</span>
        </div>
        <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 text-sm font-medium">
          <a href="#agents" className="hover:text-slate-300">Agents</a>
          <a href="#architecture" className="hover:text-slate-300">Architecture</a>
          <a href="#security" className="hover:text-slate-300">Security</a>
          <a href="#faq" className="hover:text-slate-300">FAQ</a>
        </div>
        <a href="https://github.com" className="hover:text-slate-300 flex items-center space-x-2 p-2 rounded-full hover:bg-slate-900 transition-colors">
          <Code className="w-6 h-6" />
        </a>
      </footer>
    </div>
  );
}
