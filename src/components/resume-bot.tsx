"use client"

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Minus, Square, X, Terminal, Send } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'bot'
  content: string
  timestamp: Date
}

const exampleQueries = [
  "Tell me about Vikas's experience with AI",
  "What programming languages does he know?",
  "What projects has he worked on?",
  "Describe his technical skills",
  "What's his educational background?"
]

export default function ResumeBot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: 'Welcome to VikasTerminal v1.0.0\nType your questions about Vikas or try one of the example queries below.',
      timestamp: new Date()
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [showCursor, setShowCursor] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Blinking cursor effect
  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setShowCursor(prev => !prev)
    }, 500)
    return () => clearInterval(cursorInterval)
  }, [])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isTyping) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsTyping(true)

    // Simulate bot response with typing delay
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: generateBotResponse(inputValue),
        timestamp: new Date()
      }
      setMessages(prev => [...prev, botResponse])
      setIsTyping(false)
    }, 1000 + Math.random() * 2000)
  }

  const generateBotResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase()
    
    if (lowerQuery.includes('ai') || lowerQuery.includes('artificial intelligence')) {
      return 'Vikas has extensive experience with AI and machine learning technologies. He has worked on various AI projects including natural language processing, computer vision, and predictive analytics. His expertise spans TensorFlow, PyTorch, and cloud-based AI services.'
    }
    
    if (lowerQuery.includes('programming') || lowerQuery.includes('languages')) {
      return 'Vikas is proficient in multiple programming languages including:\n• Python - Advanced level\n• JavaScript/TypeScript - Expert level\n• Java - Intermediate level\n• Go - Intermediate level\n• SQL - Advanced level\n\nHe also has experience with frameworks like React, Next.js, Django, and FastAPI.'
    }
    
    if (lowerQuery.includes('projects')) {
      return 'Vikas has worked on numerous projects including:\n• E-commerce platforms with AI-powered recommendations\n• Real-time chat applications\n• Data visualization dashboards\n• Machine learning pipelines\n• Microservices architectures\n\nEach project demonstrates his full-stack capabilities and attention to scalable solutions.'
    }
    
    if (lowerQuery.includes('skills') || lowerQuery.includes('technical')) {
      return 'Technical Skills Overview:\n• Frontend: React, Next.js, TypeScript, Tailwind CSS\n• Backend: Python, Node.js, Django, FastAPI\n• Databases: PostgreSQL, MongoDB, Redis\n• Cloud: AWS, Google Cloud, Docker, Kubernetes\n• AI/ML: TensorFlow, PyTorch, Scikit-learn\n• DevOps: CI/CD, Git, Linux, Monitoring'
    }
    
    if (lowerQuery.includes('education')) {
      return 'Education & Certifications:\n• Computer Science degree with focus on AI/ML\n• Multiple cloud certifications (AWS, GCP)\n• Continuous learning through online courses\n• Active contributor to open-source projects\n• Regular participant in tech conferences and workshops'
    }
    
    return 'I have information about Vikas\'s experience, skills, projects, and background. Try asking about his:\n• Programming languages and technologies\n• AI and machine learning experience\n• Project portfolio\n• Technical skills\n• Education and certifications\n\nOr use one of the example queries below.'
  }

  const handleExampleQuery = (query: string) => {
    setInputValue(query)
    inputRef.current?.focus()
  }

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  return (
    <div className="bg-background">
      <div className="max-w-4xl mx-auto p-6">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-card border border-border rounded-lg overflow-hidden shadow-2xl"
        >
          {/* Terminal Header */}
          <div className="bg-secondary px-4 py-3 flex items-center justify-between border-b border-border">
            <div className="flex items-center space-x-2">
              <Terminal className="w-4 h-4 text-primary" />
              <span className="font-mono text-sm text-foreground">vikas@terminal:~$</span>
            </div>
            <div className="flex items-center space-x-2">
              <button className="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 transition-colors">
                <Minus className="w-2 h-2 text-yellow-800 mx-auto" />
              </button>
              <button className="w-3 h-3 rounded-full bg-green-500 hover:bg-green-400 transition-colors">
                <Square className="w-2 h-2 text-green-800 mx-auto" />
              </button>
              <button className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 transition-colors">
                <X className="w-2 h-2 text-red-800 mx-auto" />
              </button>
            </div>
          </div>

          {/* Terminal Content */}
          <div className="h-96 overflow-y-auto p-4 space-y-4 font-mono text-sm">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-1"
                >
                  <div className="flex items-center space-x-2 text-muted-foreground">
                    <span className="text-primary">
                      {message.type === 'user' ? '┌─[user@terminal]' : '┌─[bot@terminal]'}
                    </span>
                    <span className="text-xs">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>
                  <div className="pl-4 border-l-2 border-primary/30">
                    <span className="text-primary">
                      {message.type === 'user' ? '└─$ ' : '└─> '}
                    </span>
                    <span className="text-foreground whitespace-pre-wrap">
                      {message.content}
                    </span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Typing Indicator */}
            {isTyping && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center space-x-2 text-muted-foreground"
              >
                <span className="text-primary">┌─[bot@terminal]</span>
                <span className="text-xs">typing...</span>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-border p-4 space-y-4">
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <span className="text-primary font-mono text-sm">vikas@terminal:~$</span>
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  disabled={isTyping}
                  className="w-full bg-transparent text-foreground font-mono text-sm outline-none disabled:opacity-50"
                  placeholder="Type your question..."
                  autoComplete="off"
                />
                {showCursor && (
                  <span className="absolute right-0 top-0 h-full w-2 bg-primary animate-pulse" />
                )}
              </div>
              <button
                type="submit"
                disabled={!inputValue.trim() || isTyping}
                className="p-2 text-primary hover:bg-primary/10 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>

            {/* Example Queries */}
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground font-mono">Example queries:</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {exampleQueries.map((query, index) => (
                  <button
                    key={index}
                    onClick={() => handleExampleQuery(query)}
                    disabled={isTyping}
                    className="text-left p-2 rounded bg-secondary/50 hover:bg-secondary text-xs font-mono text-foreground/70 hover:text-foreground transition-colors disabled:opacity-50"
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}