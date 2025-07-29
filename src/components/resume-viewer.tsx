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
    }, delay + currentIndex * 1)

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
    description: string[]
    technologies: string[]
    link?: string
  }[]
}

const mockResumeData: ResumeData = {
  education: [
    {
      degree: "Bachelor of Engineering",
      institution: "University of Mumbai",
      year: "2022-2026",
      details: ["GPA: 9.72/10.00", "Major: Information Technology", "Relevant Coursework: Data Structures, Algorithms, Database Systems"]
    }
  ],
  experience: [
    {
      position: "Ai Backend Engineer Intern",
      company: "Vighnotech",
      period: "Aug 2024- Jan 2025",
      responsibilities: [
        "• Collaborated with cross-functional teams to implement AI in IMS, CRM, and mobile applications.",
        "• Developed custom RAG-based applications from scratch using Scrapy and FastAPI, storing data in a vector database for optimized retrieval.",
        "• Built a MultiAgent system for chatbot applications, integrating multiple functionalities into a single solution.",
        "• Enhanced system performance and scalability by optimizing backend architecture and data pipelines."
      ]
    },
    {
      position: "Software Engineer Intern",
      company: "Jurisphere",
      period: "Mar 24 - Apr 24",
      responsibilities: [
        "• Improved RAG-augmented AI capabilities by refining data curation and processing pipelines.",
        "• Utilized Scrapy and other Python frameworks to scrape dynamic and static websites efficiently.",
        "• Automated daily report generation, enhancing team productivity and data insights.",
        "• Explored startup dynamics and contributed to enhancing AI-driven workflows."
      ]
    }
  ],
  skills: [
    {
      category: "Programming Languages",
      items: ["Python","C++", "JavaScript", "TypeScript", "Rust" ]
    },
    {
      category: "Frameworks & Tools",
      items: ["Django", "Flask", "FastAPI", "Machine Learning", "Artificial Intelligence", "GenAI", "React", "Node.js", "Docker", "AWS", "Git", "Scrapy"]
    },
    {
      category: "Databases",
      items: ["PostgreSQL", "MongoDB", "MySQL", "Vector DB"]  
    }
  ],
  projects: [
    {
      name: "Insightify",
      description: [
        "• Secured 1st place in DEVQUEST Hackathon, demonstrating exceptional problem-solving skills and innovation.\n",
        "• Curated and refined raw data for seamless front-end integration.\n",
        "• Led project presentations with a focus on strategic vision and technical depth.\n"
      ],
      technologies: ["Python"],
      link: "https://github.com/vaxad/Insightify"
    },
    {
      name: "Morpheus",
      description: [
        "• Developed a Google Forms-like application using Django Templates and DRF, ensuring modular and scalable code.\n",
        "• Created a structured folder system to enhance maintainability and modularity.\n",
        "• Enabled seamless form creation, submission, and analysis through an intuitive user interface.\n"
      ],
      technologies: ["Django", "Python", "sqlite"],
      link: "https://github.com/vickypedia-12/morpheus"
    },
    {
      name: "Study Rag",
      description: [
        "• Developed a RAG-based application to process and analyze study materials such as DOCs, PPTs, and PDFs.\n",
        "• Empowered 50+ active users to upload documents and generate comprehensive answers from the parsed questions.\n",
        "• Optimized document parsing using LangChain, streamlined backend with FastAPI, and enhanced user experience through Streamlit.\n"
      ],
      technologies: ["Python", "vector DB", "LangChain", "FastAPI", "streamlit"],
      link: "https://github.com/vickypedia-12/study-rag"
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
    const link = document.createElement("a");
    link.href = "/resume.pdf";
    link.download = "resume.pdf";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

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
    <div className="min-h-screen bg-background p-3 md:p-4 font-mono pt-16 lg:pt-0">
      <div className="max-w-4xl mx-auto">
        <Card className="bg-card border-border shadow-2xl bg-[#181a1b]">
          {/* Terminal Header */}
          <div className="border-b border-border bg-secondary p-3 md:p-4">
            <div className="flex items-center gap-1 md:gap-2">
              <div className="flex gap-1 md:gap-2">
                <div className="w-2.5 h-2.5 md:w-3 md:h-3 rounded-full bg-red-500"></div>
                <div className="w-2.5 h-2.5 md:w-3 md:h-3 rounded-full bg-yellow-500"></div>
                <div className="w-2.5 h-2.5 md:w-3 md:h-3 rounded-full bg-green-500"></div>
              </div>
              <div className="ml-2 md:ml-4 flex items-center text-primary overflow-hidden">
                <Terminal className="w-3 h-3 md:w-4 md:h-4 mr-1 md:mr-2 flex-shrink-0" style={{ color: "hsl(142 76% 56%)" }} />
                <span className="text-foreground text-xs md:text-sm truncate">vikas@portfolio</span>
                <span className="text-muted-foreground hidden sm:inline">:</span>
                <span className="text-primary hidden sm:inline" style={{ color: "hsl(142 76% 56%)" }}>~/resume</span>
                <span className="text-foreground">$</span>
                <BlinkingCursor />
              </div>
            </div>
          </div>


          {/* Resume Content */}
          <div className="p-3 md:p-6 space-y-4 md:space-y-6">
            {/* Header */}
            <div className="border-b border-border pb-3 md:pb-4">
              <div className="flex items-center gap-1 md:gap-2 mb-2">
                <User className="w-4 h-4 md:w-5 md:h-5 text-primary" style={{ color: "hsl(142 76% 56%)" }}/>
                <h1 className="text-xl md:text-2xl font-bold text-foreground">Vikas Mourya</h1>
              </div>
              <p className="text-primary text-sm md:text-lg" style={{ color: "hsl(142 76% 56%)" }}>$ whoami</p>
              <p className="text-muted-foreground ml-3 md:ml-4 opacity-60 text-sm md:text-base" >Software Engineer | Full-Stack Developer | Python Developer</p>
              <div className="mt-2 space-y-1 text-xs md:text-sm text-muted-foreground ml-3 md:ml-4 opacity-60">
                <p>📧 vikasmourya54321@gmail.com</p>
                <p>🌐 https://linkedin.com/in/vickypedia12</p>
                <p>💻 https://github.com/vickypedia-12</p>
              </div>
            </div>

            {/* Education Section */}
            <div className="space-y-2 md:space-y-3">
              <button
                onClick={() => toggleSection('education')}
                className="flex items-center gap-1 md:gap-2 w-full text-left hover:text-primary transition-colors text-sm md:text-base"
              >
                {expandedSections.includes('education') ? (
                  <ChevronDown className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                )}
                {getSectionIcon('education')}
                <span className="text-primary truncate"  style={{color: "hsl(142 76% 56%)"}}>./education</span>
                <span className="text-muted-foreground opacity-60 hidden sm:inline">--list</span>
              </button>
              
              {expandedSections.includes('education') && (
                <div className="ml-4 md:ml-8 space-y-3 md:space-y-4">
                  {mockResumeData.education.map((edu, index) => (
                    <div key={index} className="border-l-2 border-primary pl-3 md:pl-4">
                      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-1">
                        <div className="flex-1">
                          <h3 className="font-semibold text-foreground text-sm md:text-base">{edu.degree}</h3>
                          <p className="text-primary text-sm" style={{color: "hsl(142 76% 56%)"}}>{edu.institution}</p>
                        </div>
                        <span className="text-muted-foreground text-xs md:text-sm opacity-60">{edu.year}</span>
                      </div>
                      <ul className="mt-2 space-y-1 text-xs md:text-sm text-muted-foreground opacity-60">
                        {edu.details.map((detail, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-primary flex-shrink-0 mt-0.5">•</span>
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
            <div className="space-y-2 md:space-y-3">
              <button
                onClick={() => toggleSection('experience')}
                className="flex items-center gap-1 md:gap-2 w-full text-left hover:text-primary transition-colors text-sm md:text-base"
              >
                {expandedSections.includes('experience') ? (
                  <ChevronDown className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                )}
                {getSectionIcon('experience')}
                <span className="text-primary truncate"  style={{color: "hsl(142 76% 56%)"}}>./experience</span>
                <span className="text-muted-foreground opacity-60 hidden sm:inline">--show-all</span>
              </button>
              
              {expandedSections.includes('experience') && (
                <div className="ml-4 md:ml-8 space-y-3 md:space-y-4">
                  {mockResumeData.experience.map((exp, index) => (
                    <div key={index} className="border-l-2 border-primary pl-3 md:pl-4">
                      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-1">
                        <div className="flex-1">
                          <h3 className="font-semibold text-foreground text-sm md:text-base">{exp.position}</h3>
                          <p className="text-primary text-sm" style={{color: "hsl(142 76% 56%)"}}>{exp.company}</p>
                        </div>
                        <span className="text-muted-foreground text-xs md:text-sm opacity-60">{exp.period}</span>
                      </div>
                      <ul className="mt-2 space-y-1 text-xs md:text-sm text-muted-foreground opacity-60">
                        {exp.responsibilities.map((resp, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <TypewriterText text={resp} isVisible={expandedSections.includes('experience')} delay={idx * 2} />
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Skills Section */}
            <div className="space-y-2 md:space-y-3">
              <button
                onClick={() => toggleSection('skills')}
                className="flex items-center gap-1 md:gap-2 w-full text-left hover:text-primary transition-colors text-sm md:text-base"
              >
                {expandedSections.includes('skills') ? (
                  <ChevronDown className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                )}
                {getSectionIcon('skills')}
                <span className="text-primary truncate" style={{color: "hsl(142 76% 56%)"}}>./skills</span>
                <span className="text-muted-foreground opacity-60 hidden sm:inline">--verbose</span>
              </button>
              
              {expandedSections.includes('skills') && (
                <div className="ml-4 md:ml-8 space-y-3 md:space-y-4">
                  {mockResumeData.skills.map((skillGroup, index) => (
                    <div key={index} className="border-l-2 border-primary pl-3 md:pl-4">
                      <h3 className="font-semibold text-foreground mb-2 text-sm md:text-base">{skillGroup.category}</h3>
                      <div className="flex flex-wrap gap-1.5 md:gap-2">
                        {skillGroup.items.map((skill, idx) => (
                          <span key={idx} className="bg-[#181a1b] text-foreground px-2 py-1 rounded text-xs md:text-sm border border-border">
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
            <div className="space-y-2 md:space-y-3">
              <button
                onClick={() => toggleSection('projects')}
                className="flex items-center gap-1 md:gap-2 w-full text-left hover:text-primary transition-colors text-sm md:text-base"
              >
                {expandedSections.includes('projects') ? (
                  <ChevronDown className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
                )}
                {getSectionIcon('projects')}
                <span className="text-primary truncate"  style={{color: "hsl(142 76% 56%)"}}>./projects</span>
                <span className="text-muted-foreground opacity-60 hidden sm:inline">--recent</span>
              </button>
              
              {expandedSections.includes('projects') && (
                <div className="ml-4 md:ml-8 space-y-3 md:space-y-4">
                  {mockResumeData.projects.map((project, index) => (
                    <div key={index} className="border-l-2 border-primary pl-3 md:pl-4">
                      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-1">
                        <h3 className="font-semibold text-foreground text-sm md:text-base flex-1" style={{color: "hsl(142 76% 56%)"}}>{project.name}</h3>
                        {project.link && (
                          <a href={project.link} target="_blank" rel="noopener noreferrer" className="text-primary text-xs md:text-sm hover:underline self-start" style={{color: "hsl(142 76% 56%)"}}>
                            view →
                          </a>
                        )}
                      </div>
                      <ul className="mt-2 space-y-1 text-xs md:text-sm text-muted-foreground opacity-60">
                        {project.description.map((line: string, idx: number) => (
                          <li key={idx} className="flex items-start gap-2">
                            <TypewriterText
                              text={line}
                              isVisible={expandedSections.includes('projects')}
                              delay={idx * 2}
                            />
                          </li>
                        ))}
                      </ul>
                      <div className="mt-2 flex flex-wrap gap-1.5 md:gap-2">
                        {project.technologies.map((tech, idx) => (
                          <span key={idx} className="bg-secondary text-primary px-1.5 md:px-2 py-0.5 md:py-1 rounded text-xs border border-border">
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
            <div className="pt-4 md:pt-6 border-t border-border"  style={{color: "hsl(142 76% 56%)"}}>
              <Button
                onClick={handleDownload}
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-mono w-full sm:w-auto text-sm md:text-base"
              >
                <Download className="w-3 h-3 md:w-4 md:h-4 mr-1 md:mr-2"/>
                <span className="hidden sm:inline">$ curl -O resume.pdf</span>
                <span className="sm:hidden">Download Resume</span>
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}