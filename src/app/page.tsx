"use client"

import { useState, useEffect } from "react"
import SidebarNav from "@/components/sidebar-nav"
import HeroSection from "@/components/hero-section"
import AgentDeck from "@/components/agent-deck"
import ExperimentsSection from "@/components/experiments-section"
import ResumeViewer from "@/components/resume-viewer"
import ResumeBot from "@/components/resume-bot"

export default function Portfolio() {
  const [activeSection, setActiveSection] = useState("hero")
  const [currentView, setCurrentView] = useState("hero")

  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = `
      :root {
        --sidebar-background: #000000ff;
        --sidebar-border: #fffcfc5d;
        --sidebar-foreground: #ffffff;
        --sidebar-primary: hsl(217, 91%, 60%);
        --sidebar-primary-foreground: #ffffff;
        --sidebar-primary-glow:  hsl(217, 91%, 70%);
        --sidebar-accent: hsl(217, 91%, 60%);
        --sidebar-accent-foreground: hsl(210, 11%, 4%);
        --font-mono: 'JetBrains Mono', monospace;
      }
    `
    document.head.appendChild(style)
    return () => { document.head.removeChild(style) }
  }, [])

  const handleSectionClick = (sectionId: string) => {
    setActiveSection(sectionId)
    setCurrentView(sectionId)
  }

  const renderCurrentView = () => {
    switch (currentView) {
      case "hero":
        return <HeroSection />
      case "agent-deck":
        return <AgentDeck />
      case "experiments":
        return <ExperimentsSection />
      case "resume":
        return <ResumeViewer />
      case "chat":
        return <ResumeBot />
      default:
        return <HeroSection />
    }
  }

  return (
    <div className="flex min-h-screen bg-[#0a0a0a] text-white">
      <SidebarNav 
        activeSection={activeSection} 
        onSectionClick={handleSectionClick}
      />
      
      <main className="flex-1 ml-[280px] overflow-y-auto">
        {renderCurrentView()}
      </main>
    </div>
  )
}