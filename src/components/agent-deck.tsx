"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Code2 } from "lucide-react"

interface Project {
  id: string
  title: string
  description: string
  techStack: string[]
  icon: React.ReactNode
  link: string
}

const projects: Project[] = [
  {
    id: "ai-assistant",
    title: "Study RAG Assistant",
    description: "A Retrieval-Augmented Generation (RAG)-based assistant designed to help students with personalized learning, automated summarization, and context-aware Q&A from their study materials.",
    techStack: ["LangChain", "Python", "FastAPI", "Vector DB", "OpenAI API"],
    icon: <Code2 className="w-6 h-6" />,
    link: "https://github.com/vickypedia-12/StudyRag"
  },
  {
    id: "automation-engine",
    title: "Inventory Management System",
    description: "Spring Boot and PostgreSQL-based inventory system for restaurants, linking ingredients, dishes, and recipes with automated stock management.",
    techStack: ["Spring Boot", "Java", "PostgreSQL", "Docker", "REST APIs"],
    icon: <Code2 className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "content-generator",
    title: "Legal RAG Assistant",
    description: "A legal-focused Retrieval-Augmented Generation system to help law students and professionals quickly retrieve case studies, summarize documents, and provide AI-assisted insights.",
    techStack: ["LangChain", "Python", "FastAPI", "Vector DB", "OpenAI API"],
    icon: <Code2 className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "data-processor",
    title: "Morpheus – Google Forms Clone",
    description: "A Google Forms-like application built with Django, featuring dynamic form creation, API-driven responses, and an admin dashboard for analytics.",
    techStack: ["Django", "DRF", "Python", "PostgreSQL", "Bootstrap"],
    icon: <Code2 className="w-6 h-6" />,
    link: "https://github.com/vickypedia-12/morpheus"
  },
  {
    id: "code-assistant",
    title: "Sylph Search Engine",
    description: "Custom-built search engine project with web crawling, indexing, and searching capabilities, designed as part of an AOA project.",
    techStack: ["Python", "Scrapy", "FastAPI", "PostgreSQL", "Data Structures"],
    icon: <Code2 className="w-6 h-6" />,
    link: "https://github.com/vickypedia-12/sylph"
  },
  {
    id: "chat-interface",
    title: "Valentine Matchmaking App",
    description: "A Django-based matchmaking system for college students during Valentine's Week, with dynamic profile matching and recommendations.",
    techStack: ["Django", "Python", "SQLite", "Bootstrap", "jQuery"],
    icon: <Code2 className="w-6 h-6" />,
    link: "https://github.com/vickypedia-12/Valentine-hearts"
  }
]

export default function AgentDeck() {
  return (
    <section id="agent-deck" className="bg-background py-16 px-8">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl font-bold text-foreground mb-4">
            Project <span style={{ color: "hsl(217, 91%, 60%)" }}>Deck</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto" style={{ color: "hsla(0, 0%, 37%, 1.00)" }}>
            A collection of intelligent agents and tools I&apos;ve built to solve real-world problems
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <motion.div
              key={project.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ scale: 1.02 }}
              className="group"
            >
              <Card className=" bg-[#181a1b] border-border h-full relative overflow-hidden shadow-none group-hover:shadow-[0_0_24px_6px_hsl(217,91%,60%,0.35)] ">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-3 mb-3"> 
                    <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors duration-300">
                      {project.icon}
                    </div>
                    <CardTitle className="text-xl text-card-foreground group-hover:text-primary transition-colors duration-300 ">
                      {project.title}
                    </CardTitle>
                  </div>
                  
                  <CardDescription className="text-muted-foreground leading-relaxed">
                    {project.description}
                  </CardDescription>
                </CardHeader>

                <CardContent className="pt-0">
                  <div className="flex flex-wrap gap-2 mb-6" style={{color:"hsl(142 76% 56%)"}}>
                    {project.techStack.map((tech, techIndex) => (
                      <Badge
                        key={techIndex}
                        variant="secondary"
                        className=" text-primary hover:bg-primary hover:text-primary-foreground transition-colors duration-300 text-xs px-2 py-1 border-[#22ff88] rounded-full"
                      >
                        {tech}
                      </Badge>
                    ))}
                  </div>

                  <Button
                    variant="ghost"
                    className="w-full justify-between text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors duration-300"
                    asChild
                  >
                    <a href={project.link}>
                      View Project
                      <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-300" />
                    </a>
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}