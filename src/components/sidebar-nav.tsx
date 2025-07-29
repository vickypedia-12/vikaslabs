"use client";

import { useState } from "react";
import {
  ChevronRight,
  House,
  Zap,
  Code,
  FileText,
  User,
  Github,
  Linkedin,
  Mail,
} from "lucide-react";

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  href: string;
}

const navigationItems: NavItem[] = [
  {
    id: "hero",
    label: "Home",
    icon: <House className="w-5 h-6" />,
    href: "#hero",
  },
  {
    id: "agent-deck",
    label: "Project Deck",
    icon: <Zap className="w-5 h-6" />,
    href: "#agent-deck",
  },
  {
    id: "experiments",
    label: "Experiments",
    icon: <Code className="w-5 h-6" />,
    href: "#experiments",
  },
  {
    id: "resume",
    label: "Resume",
    icon: <FileText className="w-5 h-6" />,
    href: "#resume",
  },
  {
    id: "chat",
    label: "Chat",
    icon: <User className="w-5 h-6" />,
    href: "#chat",
  },
];

const socialLinks = [
  {
    id: "github",
    icon: <Github className="w-4 h-4" />,
    href: "https://github.com/vickypedia-12",
    label: "GitHub",
  },
  {
    id: "linkedin",
    icon: <Linkedin className="w-4 h-4" />,
    href: "https://linkedin.com/in/vickypedia12",
    label: "LinkedIn",
  },
  {
    id: "email",
    icon: <Mail className="w-4 h-4" />,
    href: "mailto:vikasmourya54321@gmail.com",
    label: "Email",
  },
];

interface SidebarNavProps {
  activeSection?: string;
  onSectionClick?: (sectionId: string) => void;
  open?: boolean; // <-- add this
  onClose?: () => void; // <-- add this
}

export default function SidebarNav({
  activeSection,
  onSectionClick,
  open = false,
  onClose,
}: SidebarNavProps) {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  const handleNavClick = (item: NavItem) => {
    if (onSectionClick) {
      onSectionClick(item.id);
    }
    if (onClose) {
      onClose(); // close sidebar on mobile after click
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      
      <div
        className={`
          fixed left-0 top-0 h-full w-[280px] bg-[var(--sidebar-background)] border-r border-[var(--sidebar-border)] backdrop-blur-sm bg-opacity-95 z-50 flex flex-col
          transition-transform duration-300 ease-in-out
          ${open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Close button for mobile only */}
        {open && (
          <button
            className="absolute top-4 right-4 p-1 rounded-md bg-[#181a1b] text-white lg:hidden z-10"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}

        {/* Header */}
        <div className="p-6 border-b border-[var(--sidebar-border)]">
          <div className="flex items-center gap-3">
            {/* Blue icon box */}
            <span
              className="flex items-center justify-center w-12 h-11 rounded-2xl"
              style={{
                background: "hsl(217,91%,60%)",
              }}
            >
              <span
                className="font-mono text-3xl"
                style={{
                  color: "hsl(210,11%,4%)",
                  fontWeight: 400,
                  letterSpacing: "0.05em",
                }}
              >
                &gt;_
              </span>
            </span>
            {/* Texts */}
            <div>
              <h1 className="text-xl font-bold font-mono leading-tight">
                vikas.lab
              </h1>
              <h3 className="text-base font-mono text-white/50 opacity-80 leading-tight">
                AI Engineer
              </h3>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 overflow-y-auto">
          <div className="space-y-1">
            {navigationItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleNavClick(item)}
                onMouseEnter={() => setHoveredItem(item.id)}
                onMouseLeave={() => setHoveredItem(null)}
                className={`
                  w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-md transition-all duration-200
                  border border-transparent bg-transparent
                  ${
                    activeSection === item.id
                      ? "bg-[#181a1b] text-[var(--sidebar-primary-foreground)] bg-[#181a1b] shadow-[0_0_8px_2px_hsl(217,91%,60%,0.25)] border-blue-400"
                      : "bg-[#16181a] text-[var(--sidebar-foreground)] hover:border-[#222] hover:bg-[#181a1b]"
                  }
                  ${hoveredItem === item.id ? "translate-x-1" : ""}
                `}
              >
                <ChevronRight
                  className={`
                    w-3 h-3 transition-transform duration-200
                    ${activeSection === item.id ? "rotate-90" : ""}
                  `}
                />
                {item.icon}
                <span className="font-[var(--font-mono)] text-lg">{item.label}</span>
              </button>
            ))}
          </div>
        </nav>

        {/* Footer with social links */}
        <div className="p-6 border-t border-[var(--sidebar-border)]">
          <div className="flex items-center justify-center gap-4">
            {socialLinks.map((link) => (
              <a
                key={link.id}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-md text-[var(--sidebar-foreground)] hover:bg-[var(--sidebar-accent)] hover:text-[var(--sidebar-accent-foreground)] transition-colors duration-200"
                aria-label={link.label}
              >
                {link.icon}
              </a>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
