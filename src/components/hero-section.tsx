"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

export default function HeroSection() {
  const scrollToAgentDeck = () => {
    const agentDeckSection = document.getElementById('agent-deck')
    if (agentDeckSection) {
      agentDeckSection.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section className="min-h-screen bg-background flex items-center justify-center px-4 relative overflow-hidden">
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-card/20 pointer-events-none" />
      
      {/* Content */}
      <div className="relative z-10 text-center max-w-4xl mx-auto">
        {/* Main heading */}
        <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-foreground mb-6 leading-tight">
          Hi, I'm Vikas — <br className="hidden sm:block" />
          <span className="text-primary">Python & AI Engineer</span>
        </h1>
        
        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
          Specializing in backend development, AI agents, LLM tools, and intelligent automation systems
        </p>
        
        {/* Brief paragraph */}
        <p className="text-lg text-muted-foreground mb-12 max-w-xl mx-auto">
          Final-year engineering student building the future with AI agents, backend systems, and intelligent automation.
        </p>
        
        {/* CTA Button */}
        <Button 
          onClick={scrollToAgentDeck}
          size="lg"
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-8 py-6 text-lg group transition-all duration-300"
        >
          View My Work
          <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
        </Button>
      </div>
    </section>
  )
}