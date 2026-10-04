"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  LineChart, 
  Briefcase, 
  Settings, 
  MessageSquare,
  Bot
} from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/briefs", label: "Market Briefs", icon: MessageSquare },
  { href: "/portfolio", label: "Portfolio", icon: Briefcase },
  { href: "/analytics", label: "Analytics", icon: LineChart },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-slate-900 text-white min-h-screen flex flex-col fixed left-0 top-0 z-40">
      <div className="p-6 flex items-center space-x-3 border-b border-slate-800">
        <div className="bg-blue-600 p-2 rounded-xl">
          <Bot className="w-6 h-6 text-white" />
        </div>
        <h1 className="text-xl font-bold tracking-tight">FinAgent OS</h1>
      </div>
      
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? "bg-blue-600/20 text-blue-400"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-800">
        <Link
          href="/settings"
          className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
            pathname === "/settings"
              ? "bg-blue-600/20 text-blue-400"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          }`}
        >
          <Settings className="w-5 h-5" />
          <span className="font-medium">Settings</span>
        </Link>
      </div>
    </aside>
  );
}

