"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, Pause } from "lucide-react"
import React, { useEffect, useState } from "react";


  const TITLES = ["Python Developer", "AI Engineer"];
  const TYPING_SPEED = 80;
  const PAUSE = 1500;

export default function HeroSection() {
  const [displayed, setDisplayed] = useState("");
  const [titleIdx, setTitleIdx] = useState(0);
  const [typing, setTyping] = useState(true);

  useEffect(() => {
    let timeout: NodeJS.Timeout;

    if (typing) {
      if (displayed.length < TITLES[titleIdx].length) {
        timeout = setTimeout(() => {
          setDisplayed(TITLES[titleIdx].slice(0, displayed.length + 1));
        }, TYPING_SPEED);
      } else {
        timeout = setTimeout(() => setTyping(false), PAUSE);
      }
    } else {
      if (displayed.length > 0) {
        timeout = setTimeout(() => {
          setDisplayed(TITLES[titleIdx].slice(0, displayed.length - 1));
        }, TYPING_SPEED / 2);
      } else {
        timeout = setTimeout(() => {
          setTitleIdx((titleIdx + 1) % TITLES.length);
          setTyping(true);
        }, 400);
      }
    }

    return () => clearTimeout(timeout);
  }, [displayed, typing, titleIdx]);

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
    <div className="relative z-10 flex flex-col items-center max-w-5xl mx-auto">
      {/* Terminal-like window */}
      <div className="bg-[#111315] rounded-2xl shadow-2xl border border-[#222] w-360 max-w-4xl mx-auto overflow-hidden"
      style={{
        boxShadow: "0 0 32px 4px hsl(217,91%,60%,0.25), 0 0 0 2px hsl(217,91%,60%,0.15)",
      }}>
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-2 bg-[#181a1b] border-b border-[#222]">
          <div className="flex items-center gap-2">
            {/* Red, Yellow, Green buttons */}
            <span className="w-3 h-3 rounded-full bg-red-500 inline-block" />
            <span className="w-3 h-3 rounded-full bg-yellow-400 inline-block" />
            <span className="w-3 h-3 rounded-full bg-green-500 inline-block" />
          </div>
          <span className="text-xs font-mono text-gray-400">~/vikas-lab</span>
          <span /> {/* Spacer for symmetry */}
        </div>
        {/* Terminal content */}
        <div className="px-8 py-6 md:py-8 flex flex-col items-center">
          {/* Optional: Terminal prompt */}
          <div className="w-full text-left mb-4">
            <span className="text-green-400 font-mono">$</span>
            <span className="text-gray-300 font-mono ml-2">whoami</span>
          </div>
          {/* Main heading */}
          <h1 className="text-2xl md:text-4xl lg:text-5xl font-bold text-foreground mb-4 leading-tight font-mono text-center">
            Hi, I&apos;m Vikas — <br className="hidden sm:block" />
            <span className="text-primary" style={{ color: "hsl(217,91%,60%)" }}>
              {displayed}&nbsp;<span className="animate-pulse">|</span>
            </span>
          </h1>
        </div>
      </div>
      {/* Subtitle OUTSIDE the terminal box */}
      <p className="text-lg text-muted-foreground w-[760px] mt-8 mb-8 mx-auto text-center">
        Final-year engineering student building the future with{" "}
        <span style={{ color: "#60a5fa", fontWeight: 600 }}>backend development</span>,{" "}
        <span style={{ color: "#60a5fa", fontWeight: 600 }}>AI agents</span>,{" "}
        <span style={{ color: "#60a5fa", fontWeight: 600 }}>LLM tools</span>, and{" "}
        <span style={{ color: "#60a5fa", fontWeight: 600 }}>intelligent automation systems</span>
      </p>
      {/* CTA Button */}
      <div className="flex flex-row gap-4 justify-center">
      <Button
        onClick={scrollToAgentDeck}
        size="lg"
        className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-8 py-6 text-lg group transition-all duration-300 shadow-[0_0_16px_2px_hsl(217,91%,60%,0.35)]"
      >
        View My Work
        <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
      </Button>
      <a
        href="/resume.pdf"
        download
        className="inline-block"
      >
        <Button
          variant="outline"
          size="lg"
          className="font-semibold px-8 py-6 text-lg shadow-[0_0_16px_2px_hsl(217,91%,60%,0.35)]"
          type="button"
        >
          Download Resume
        </Button>
      </a>
    </div>
    </div>
  </section>
)
}