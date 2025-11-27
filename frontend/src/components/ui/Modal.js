import React, { useEffect } from "react";

const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  size = "md",
  bodyClassName = "",
}) => {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const sizes = {
    sm: "max-w-md",
    md: "max-w-xl",
    lg: "max-w-3xl",
    xl: "max-w-5xl",
    full: "w-full h-full max-w-none m-0 rounded-none",
  };

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center ${
        size === "full" ? "p-0" : "p-4 sm:p-6"
      }`}
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/80 backdrop-blur-sm transition-opacity animate-fade-in"
        onClick={onClose}
      />

      {/* Modal Content */}
      <div
        className={`
          relative w-full ${sizes[size]} 
          bg-surface border border-white/10 ${
            size === "full" ? "" : "rounded-2xl"
          } shadow-2xl 
          transform transition-all animate-slide-up
          flex flex-col ${size === "full" ? "h-full" : "max-h-[90vh]"}
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <h3 className="text-xl font-heading font-semibold text-white">
            {title}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/5"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div
          className={`p-6 overflow-y-auto custom-scrollbar ${bodyClassName}`}
        >
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;
