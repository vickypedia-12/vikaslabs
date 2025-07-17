"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Download, ChevronRight, ChevronDown, Terminal, User, GraduationCap, Briefcase, Code, FolderOpen } from 'lucide-react'

const BlinkingCursor = () => (
  <span className="inline-block w-2 h-5 bg-primary ml-1 animate-pulse">
    <span className="sr-only">cursor</span>
  </span>
)

const TypewriterText = ({ text, isVisible, delay = 0 }: { text: string; isVisible: boolean; delay?: number }) => {
  const [displayedText, setDisplayedText] = useState("")
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    if (!isVisible) {
      setDisplayedText("")
      setCurrentIndex(0)
      return
    }

    const timer = setTimeout(() => {
      if (currentIndex < text.length) {
        setDisplayedText(text.slice(0, currentIndex + 1))
        setCurrentIndex(currentIndex + 1)
      }
    }, delay + currentIndex * 50)

    return () => clearTimeout(timer)
  }, [text, isVisible, currentIndex, delay])

  return <span>{displayedText}</span>
}

interface ResumeData {
  education: {
    degree: string
    institution: string
    year: string
    details: string[]
  }[]
  experience: {
    position: string
    company: string
    period: string
    responsibilities: string[]
  }[]
  skills: {
    category: string
    items: string[]
  }[]
  projects: {
    name: string
    description: string
    technologies: string[]
    link?: string
  }[]
}

const mockResumeData: ResumeData = {
  education: [
    {
      degree: "Master of Computer Science",
      institution: "Stanford University",
      year: "2021-2023",
      details: ["GPA: 3.8/4.0", "Specialization: Machine Learning", "Research: Natural Language Processing"]
    },
    {
      degree: "Bachelor of Engineering",
      institution: "Indian Institute of Technology",
      year: "2017-2021",
      details: ["GPA: 3.9/4.0", "Major: Computer Science", "Valedictorian"]
    }
  ],
  experience: [
    {
      position: "Senior Software Engineer",
      company: "Meta",
      period: "2023-Present",
      responsibilities: [
        "Lead development of ML infrastructure serving 3B+ users",
        "Architected microservices handling 100M+ requests/day",
        "Mentored 5 junior engineers and led cross-functional teams"
      ]
    },
    {
      position: "Software Engineer Intern",
      company: "Google",
      period: "Summer 2022",
      responsibilities: [
        "Developed real-time data processing pipeline",
        "Improved query performance by 40% using advanced indexing",
        "Collaborated with product teams on user-facing features"
      ]
    }
  ],
  skills: [
    {
      category: "Programming Languages",
      items: ["Python", "JavaScript", "TypeScript", "Go", "Rust", "C++"]
    },
    {
      category: "Frameworks & Tools",
      items: ["React", "Node.js", "Docker", "Kubernetes", "AWS", "TensorFlow"]
    },
    {
      category: "Databases",
      items: ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"]
    }
  ],
  projects: [
    {
      name: "AI Code Assistant",
      description: "Built an intelligent code completion tool using GPT-4 API",
      technologies: ["Python", "OpenAI API", "VS Code Extension"],
      link: "https://github.com/vikas/ai-assistant"
    },
    {
      name: "Distributed Chat System",
      description: "Real-time chat application with 99.9% uptime",
      technologies: ["Go", "WebSocket", "Redis", "PostgreSQL"],
      link: "https://github.com/vikas/chat-system"
    }
  ]
}

export default function ResumeViewer() {
  const [expandedSections, setExpandedSections] = useState<string[]>([])
  const [commandHistory, setCommandHistory] = useState<string[]>([])
  const [currentCommand, setCurrentCommand] = useState("")

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
    
    const command = expandedSections.includes(section) ? `hide ${section}` : `show ${section}`
    setCommandHistory(prev => [...prev, command])
  }

  const handleDownload = () => {
    setCommandHistory(prev => [...prev, "download resume.pdf"])
    // Mock download functionality
    console.log("Downloading resume...")
  }

  const getSectionIcon = (section: string) => {
    switch (section) {
      case 'education': return <GraduationCap className="w-4 h-4" />
      case 'experience': return <Briefcase className="w-4 h-4" />
      case 'skills': return <Code className="w-4 h-4" />
      case 'projects': return <FolderOpen className="w-4 h-4" />
      default: return <Terminal className="w-4 h-4" />
    }
  }

  return (
    <div className="min-h-screen bg-background p-4 font-mono">
      <div className="max-w-4xl mx-auto">
        <Card className="bg-card border-border shadow-2xl">
          {/* Terminal Header */}
          <div className="border-b border-border bg-secondary p-4">
            <div className="flex items-center gap-2">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <div className="ml-4 flex items-center text-primary">
                <Terminal className="w-4 h-4 mr-2" />
                <span className="text-foreground">vikas@portfolio</span>
                <span className="text-muted-foreground">:</span>
                <span className="text-primary">~/resume</span>
                <span className="text-foreground">$</span>
                <BlinkingCursor />
              </div>
            </div>
          </div>

          {/* Command History */}
          <div className="p-4 border-b border-border bg-muted/30">
            <div className="space-y-1 text-sm">
              {commandHistory.slice(-3).map((cmd, index) => (
                <div key={index} className="flex items-center text-muted-foreground">
                  <span className="text-primary">$</span>
                  <span className="ml-2">{cmd}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Resume Content */}
          <div className="p-6 space-y-6">
            {/* Header */}
            <div className="border-b border-border pb-4">
              <div className="flex items-center gap-2 mb-2">
                <User className="w-5 h-5 text-primary" />
                <h1 className="text-2xl font-bold text-foreground">Vikas Kumar</h1>
              </div>
              <p className="text-primary text-lg">$ whoami</p>
              <p className="text-muted-foreground ml-4">Senior Software Engineer | Full-Stack Developer | ML Engineer</p>
              <div className="mt-2 space-y-1 text-sm text-muted-foreground ml-4">
                <p>📧 vikas.kumar@example.com</p>
                <p>🌐 linkedin.com/in/vikas-kumar</p>
                <p>💻 github.com/vikas</p>
              </div>
            </div>

            {/* Education Section */}
            <div className="space-y-3">
              <button
                onClick={() => toggleSection('education')}
                className="flex items-center gap-2 w-full text-left hover:text-primary transition-colors"
              >
                {expandedSections.includes('education') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                {getSectionIcon('education')}
                <span className="text-primary">./education</span>
                <span className="text-muted-foreground">--list</span>
              </button>
              
              {expandedSections.includes('education') && (
                <div className="ml-8 space-y-4">
                  {mockResumeData.education.map((edu, index) => (
                    <div key={index} className="border-l-2 border-primary pl-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-semibold text-foreground">{edu.degree}</h3>
                          <p className="text-primary">{edu.institution}</p>
                        </div>
                        <span className="text-muted-foreground text-sm">{edu.year}</span>
                      </div>
                      <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                        {edu.details.map((detail, idx) => (
                          <li key={idx} className="flex items-center gap-2">
                            <span className="text-primary">•</span>
                            <TypewriterText text={detail} isVisible={expandedSections.includes('education')} delay={idx * 100} />
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Experience Section */}
            <div className="space-y-3">
              <button
                onClick={() => toggleSection('experience')}
                className="flex items-center gap-2 w-full text-left hover:text-primary transition-colors"
              >
                {expandedSections.includes('experience') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                {getSectionIcon('experience')}
                <span className="text-primary">./experience</span>
                <span className="text-muted-foreground">--show-all</span>
              </button>
              
              {expandedSections.includes('experience') && (
                <div className="ml-8 space-y-4">
                  {mockResumeData.experience.map((exp, index) => (
                    <div key={index} className="border-l-2 border-primary pl-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-semibold text-foreground">{exp.position}</h3>
                          <p className="text-primary">{exp.company}</p>
                        </div>
                        <span className="text-muted-foreground text-sm">{exp.period}</span>
                      </div>
                      <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                        {exp.responsibilities.map((resp, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-primary mt-1">•</span>
                            <TypewriterText text={resp} isVisible={expandedSections.includes('experience')} delay={idx * 150} />
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Skills Section */}
            <div className="space-y-3">
              <button
                onClick={() => toggleSection('skills')}
                className="flex items-center gap-2 w-full text-left hover:text-primary transition-colors"
              >
                {expandedSections.includes('skills') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                {getSectionIcon('skills')}
                <span className="text-primary">./skills</span>
                <span className="text-muted-foreground">--verbose</span>
              </button>
              
              {expandedSections.includes('skills') && (
                <div className="ml-8 space-y-4">
                  {mockResumeData.skills.map((skillGroup, index) => (
                    <div key={index} className="border-l-2 border-primary pl-4">
                      <h3 className="font-semibold text-foreground mb-2">{skillGroup.category}</h3>
                      <div className="flex flex-wrap gap-2">
                        {skillGroup.items.map((skill, idx) => (
                          <span key={idx} className="bg-secondary text-foreground px-2 py-1 rounded text-sm border border-border">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Projects Section */}
            <div className="space-y-3">
              <button
                onClick={() => toggleSection('projects')}
                className="flex items-center gap-2 w-full text-left hover:text-primary transition-colors"
              >
                {expandedSections.includes('projects') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                {getSectionIcon('projects')}
                <span className="text-primary">./projects</span>
                <span className="text-muted-foreground">--recent</span>
              </button>
              
              {expandedSections.includes('projects') && (
                <div className="ml-8 space-y-4">
                  {mockResumeData.projects.map((project, index) => (
                    <div key={index} className="border-l-2 border-primary pl-4">
                      <div className="flex justify-between items-start">
                        <h3 className="font-semibold text-foreground">{project.name}</h3>
                        {project.link && (
                          <a href={project.link} target="_blank" rel="noopener noreferrer" className="text-primary text-sm hover:underline">
                            view →
                          </a>
                        )}
                      </div>
                      <p className="text-muted-foreground text-sm mt-1">{project.description}</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {project.technologies.map((tech, idx) => (
                          <span key={idx} className="bg-secondary text-primary px-2 py-1 rounded text-xs border border-border">
                            {tech}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Download Button */}
            <div className="pt-6 border-t border-border">
              <Button
                onClick={handleDownload}
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-mono"
              >
                <Download className="w-4 h-4 mr-2" />
                $ curl -O resume.pdf
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}