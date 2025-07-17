"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Bot, Code, Database, Globe, MessageSquare, Zap } from "lucide-react"

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
    title: "AI Assistant Platform",
    description: "Intelligent conversational agent with multi-model support, context awareness, and custom knowledge integration.",
    techStack: ["OpenAI GPT", "LangChain", "TypeScript", "React", "Vector DB"],
    icon: <Bot className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "automation-engine",
    title: "Workflow Automation Engine",
    description: "Advanced automation system for complex business processes with AI-driven decision making and API integrations.",
    techStack: ["Python", "FastAPI", "Redis", "PostgreSQL", "Docker"],
    icon: <Zap className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "content-generator",
    title: "Content Generation Suite",
    description: "Multi-modal content creation platform leveraging cutting-edge AI models for text, images, and code generation.",
    techStack: ["Next.js", "Stable Diffusion", "Transformers", "AWS", "MongoDB"],
    icon: <Globe className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "data-processor",
    title: "Intelligent Data Processor",
    description: "Real-time data processing pipeline with ML-powered insights, anomaly detection, and predictive analytics.",
    techStack: ["Apache Kafka", "TensorFlow", "Kubernetes", "ElasticSearch", "Go"],
    icon: <Database className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "code-assistant",
    title: "Code Assistant Tool",
    description: "AI-powered development companion with intelligent code completion, review automation, and documentation generation.",
    techStack: ["VS Code API", "CodeT5", "Node.js", "GitHub API", "WebSocket"],
    icon: <Code className="w-6 h-6" />,
    link: "#"
  },
  {
    id: "chat-interface",
    title: "Conversational Interface",
    description: "Advanced chat interface with voice integration, multi-language support, and context-aware responses.",
    techStack: ["React", "WebRTC", "Socket.io", "Speech API", "NLP"],
    icon: <MessageSquare className="w-6 h-6" />,
    link: "#"
  }
]

export default function AgentDeck() {
  return (
    <section className="bg-background py-16 px-8">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl font-bold text-foreground mb-4">
            Agent Deck
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Explore our collection of intelligent AI agents and automation tools designed to enhance productivity and streamline workflows.
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
              whileHover={{ y: -8 }}
              className="group"
            >
              <Card className="bg-card border-border h-full relative overflow-hidden backdrop-blur-sm bg-opacity-90 hover:bg-opacity-100 transition-all duration-300">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors duration-300">
                      {project.icon}
                    </div>
                    <CardTitle className="text-xl text-card-foreground group-hover:text-primary transition-colors duration-300">
                      {project.title}
                    </CardTitle>
                  </div>
                  
                  <CardDescription className="text-muted-foreground leading-relaxed">
                    {project.description}
                  </CardDescription>
                </CardHeader>

                <CardContent className="pt-0">
                  <div className="flex flex-wrap gap-2 mb-6">
                    {project.techStack.map((tech, techIndex) => (
                      <Badge
                        key={techIndex}
                        variant="secondary"
                        className="bg-primary/10 text-primary hover:bg-primary hover:text-primary-foreground transition-colors duration-300 text-xs px-2 py-1"
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