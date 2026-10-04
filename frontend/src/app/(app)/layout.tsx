import { Sidebar } from "@/components/Sidebar";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Sidebar />
      <div className="flex-1 ml-64 flex flex-col min-h-screen">
        {children}
      </div>
    </>
  );
}
