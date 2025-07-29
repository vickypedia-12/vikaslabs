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

  // Auto-scroll to bottom when messages change (but not on initial load)
  useEffect(() => {
    // Only auto-scroll if there are more than 1 message (initial welcome message)
    if (messages.length > 1) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
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
  
  if( lowerQuery.includes('hello') || lowerQuery.includes('hello, how are you')|| lowerQuery.includes('hi, how are you') || lowerQuery.includes('hey, how are you')) {
    return 'Hello! I am a bot, created in vikas.lab, How can I assist you today?'
  }

  if (lowerQuery.includes('ai') || lowerQuery.includes('artificial intelligence')) {
    return 'Vikas has hands-on experience in AI and machine learning. He has worked on projects involving Retrieval-Augmented Generation (RAG), custom chatbots, multi-agent systems, and computer vision for safety monitoring. He uses tools like PyTorch, Scikit-learn, and LangChain, along with vector databases for RAG-based apps.'
  }
  
  if (lowerQuery.includes('programming') || lowerQuery.includes('languages')) {
    return 'Vikas is proficient in multiple programming languages including:\n• Python - Advanced level\n• Java - Intermediate level\n• C++ - Intermediate level\n• SQL - Advanced level\n\nHe has also worked extensively with Django, FastAPI, Spring Boot, and React.'
  }
  
  if (lowerQuery.includes('projects')) {
    return "Vikas has worked on a wide range of projects, such as:\n• A custom RAG-based chatbot system with automated data scraping\n• A violence detection and workplace safety monitoring system\n• An accessible voting platform for people with disabilities\n• A hotel and inventory management system using Spring Boot & PostgreSQL\n• Sylph Search Engine – a custom-built search engine with crawler and FastAPI backend\n\nThese projects reflect his focus on AI-driven applications and scalable backend systems."
  }
  
  if (lowerQuery.includes('skills') || lowerQuery.includes('technical')) {
    return "Technical Skills Overview:\n • Programming & Backend: Python, Django, FastAPI, Spring Boot\n • Frontend & UI: React, JavaScript, TypeScript, Tailwind CSS\n • Databases & Caching: PostgreSQL, MongoDB, Redis\n • AI & Machine Learning: PyTorch, Scikit-learn, LangChain\n • Cloud & Tools: AWS, Linux, Docker, Git\n • Others: Web scraping (Scrapy), RAG apps, and CI/CD pipelines"
  }
  
  if (lowerQuery.includes('education')) {
    return 'Education & Certifications:\n• B.E. in Computer Engineering (3rd year) at Thakur College of Engineering & Technology\n• Experience through internships (6 months) in AI-driven projects\n• Active participant in hackathons like Code for Good and GSoC preparation\n• Continuous learning in AI, backend systems, and algorithms'
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
    <div className="bg-background pt-20 lg:pt-6">
      <div className="max-w-4xl mx-auto p-3 md:p-6" >
        <motion.div 
          initial={{ opacity: 0,  y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-[#181a1b] border border-border rounded-lg overflow-hidden shadow-2xl"
        >
          {/* Terminal Header */}
          <div className="bg-secondary px-3 md:px-4 py-2 md:py-3 flex items-center justify-between border-b border-border">
            <div className="flex items-center space-x-1 md:space-x-2">
              <Terminal className="w-3 h-3 md:w-4 md:h-4 text-primary" style={{ color: "hsl(142 76% 56%)" }} />
              <span className="font-mono text-xs md:text-sm text-foreground truncate" >vikas.lab@terminal:~$</span>
            </div>
            <div className="flex items-center space-x-1 md:space-x-2">
              <button className="w-2.5 h-2.5 md:w-3 md:h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 transition-colors">
                <Minus className="w-1.5 h-1.5 md:w-2 md:h-2 text-yellow-800 mx-auto" />
              </button>
              <button className="w-2.5 h-2.5 md:w-3 md:h-3 rounded-full bg-green-500 hover:bg-green-400 transition-colors">
                <Square className="w-1.5 h-1.5 md:w-2 md:h-2 text-green-800 mx-auto" />
              </button>
              <button className="w-2.5 h-2.5 md:w-3 md:h-3 rounded-full bg-red-500 hover:bg-red-400 transition-colors">
                <X className="w-1.5 h-1.5 md:w-2 md:h-2 text-red-800 mx-auto" />
              </button>
            </div>
          </div>

          {/* Terminal Content */}
          <div className="h-80 md:h-96 overflow-y-auto p-3 md:p-4 space-y-3 md:space-y-4 font-mono text-xs md:text-sm">
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
                  <div className="flex items-center space-x-1 md:space-x-2 text-muted-foreground">
                    <span className="text-primary text-xs md:text-sm" style={{ color: "hsl(142 76% 56%)" }}>
                      {message.type === 'user' ? '┌─[user@terminal]' : '┌─[bot@terminal]'}
                    </span>
                    <span className="text-xs">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>
                  <div className="pl-3 md:pl-4 border-l-2 border-primary/30" >
                    <span className="text-primary text-xs md:text-sm" style={{ color: "hsl(142 76% 56%)" }}>
                      {message.type === 'user' ? '└─$ ' : '└─> '}
                    </span>
                    <span className="text-foreground whitespace-pre-wrap text-xs md:text-sm">
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
                className="flex items-center space-x-1 md:space-x-2 text-muted-foreground"
              >
                <span className="text-primary text-xs md:text-sm">┌─[bot@terminal]</span>
                <span className="text-xs">typing...</span>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-border p-3 md:p-4 space-y-3 md:space-y-4">
            <form onSubmit={handleSubmit} className="flex items-center space-x-1 md:space-x-2">
              <span className="text-primary font-mono text-xs md:text-sm hidden sm:inline" style={{ color: "hsl(142 76% 56%)" }}>vikas.lab@terminal:~$</span>
              <span className="text-primary font-mono text-xs sm:hidden" style={{ color: "hsl(142 76% 56%)" }}>$</span>
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  disabled={isTyping}
                  className="w-full bg-transparent text-foreground font-mono text-xs md:text-sm outline-none disabled:opacity-50"
                  placeholder="Type your question..."
                  autoComplete="off"
                />
                <span
                  className={`absolute right-0 top-0 h-full w-1 md:w-1.5 bg-[#22ff88] animate-blink`}
                  style={{ borderRadius: 2 }}
                />
              </div>
              <button
                type="submit"
                disabled={!inputValue.trim() || isTyping}
                className="p-1.5 md:p-2 text-primary hover:bg-primary/10 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                style={{ color: "hsl(142 76% 56%)" }} >
                <Send className="w-3 h-3 md:w-4 md:h-4" />
              </button>
            </form>

            {/* Example Queries */}
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground font-mono">Example queries:</div>
              <div className="grid grid-cols-1 gap-2">
                {exampleQueries.map((query, index) => (
                  <button
                    key={index}
                    onClick={() => handleExampleQuery(query)}
                    disabled={isTyping}
                    className="text-left p-2 md:p-2.5 rounded bg-secondary/50 hover:bg-secondary text-xs font-mono text-foreground/70 hover:text-foreground transition-colors disabled:opacity-50 leading-relaxed"
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