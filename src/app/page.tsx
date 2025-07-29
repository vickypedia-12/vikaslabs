"use client"

import { useState, useEffect } from "react"
import SidebarNav from "@/components/sidebar-nav"
import HeroSection from "@/components/hero-section"
import AgentDeck from "@/components/agent-deck"
import ExperimentsSection from "@/components/experiments-section"
import ResumeViewer from "@/components/resume-viewer"
import ResumeBot from "@/components/resume-bot"

export default function Portfolio() {
  const [activeSection, setActiveSection] = useState("hero");
  const [currentView, setCurrentView] = useState("hero");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      :root {
        --sidebar-background: #141414ff;
        --sidebar-border: #fffcfc5d;
        --sidebar-foreground: #ffffff;
        --sidebar-primary: hsl(217, 91%, 60%);
        --sidebar-primary-foreground: #ffffff;
        --sidebar-primary-glow:  hsl(217, 91%, 70%);
        --sidebar-accent: hsl(217, 91%, 60%);
        --sidebar-accent-foreground: hsl(210, 11%, 4%);
        --font-mono: 'JetBrains Mono', monospace;
      }
    `;
    document.head.appendChild(style);
    return () => { document.head.removeChild(style); };
  }, []);

  const handleSectionClick = (sectionId: string) => {
    setActiveSection(sectionId);
    setCurrentView(sectionId);
    setSidebarOpen(false); // close sidebar on mobile after click
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case "hero":
        return <HeroSection goToAgentDeck={() => setCurrentView("agent-deck")} />;
      case "agent-deck":
        return <AgentDeck />;
      case "experiments":
        return <ExperimentsSection />;
      case "resume":
        return <ResumeViewer />;
      case "chat":
        return <ResumeBot />;
      default:
        return <HeroSection goToAgentDeck={() => setCurrentView("agent-deck")} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0a0a0a] text-white">
      {/* Clean hamburger menu for mobile */}
      <button
        className="fixed top-4 left-4 z-50 group lg:hidden p-3"
        onClick={() => setSidebarOpen(true)}
        aria-label="Open sidebar"
      >
        <div className="flex flex-col gap-1.5 w-6">
          <div className="w-6 h-1 bg-white rounded-full transition-all duration-300 group-hover:bg-primary"></div>
          <div className="w-4 h-1 bg-white rounded-full transition-all duration-300 group-hover:bg-primary"></div>
          <div className="w-6 h-1 bg-white rounded-full transition-all duration-300 group-hover:bg-primary"></div>
        </div>
      </button>

      <SidebarNav
        activeSection={activeSection}
        onSectionClick={handleSectionClick}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex-1 lg:ml-[280px] overflow-y-auto">
        {renderCurrentView()}
      </main>
    </div>
  );
}