"use client";
import React, { useRef } from "react";
import {
  motion,
  useAnimationFrame,
  useMotionTemplate,
  useMotionValue,
  useTransform,
} from "motion/react";
import { cn } from "@/lib/utils";

export function Buttn({
  borderRadius = "1.75rem",
  text,
  onClick,
  as: Component = "button",
  containerClassName,
  borderClassName,
  duration = 3000,
  className,
  ...otherProps
}: {
  borderRadius?: string;
  text?: string;
  onClick?: () => void;
  as?: any;
  containerClassName?: string;
  borderClassName?: string;
  duration?: number;
  className?: string;
  [key: string]: any;
}) {
  return (
    <Component
      onClick={onClick}
      className={cn(
        "relative flex items-center justify-center h-16 w-40 overflow-hidden bg-transparent p-[1px] text-xl",
        containerClassName
      )}
      style={{
        borderRadius: borderRadius,
      }}
      {...otherProps}
    >
      {/* Borde animado detrás */}
      <div
        className="absolute inset-0 z-0 pointer-events-none"
        style={{ borderRadius: `calc(${borderRadius} * 0.96)` }}
      >
        <MovingBorder duration={duration} rx="30%" ry="30%">
          <div
            className={cn(
              "h-20 w-20 bg-[radial-gradient(#ffffff_40%,transparent_60%)] opacity-[0.8]",
              borderClassName
            )}
          />
        </MovingBorder>
      </div>

      {/* Contenido del botón */}
      <div
        className={cn(
          "relative flex h-full w-[97%] items-center justify-center border border-slate-800 bg-slate-900/[0.8] text-sm text-white antialiased backdrop-blur-xl font-bold",
          className
        )}
        style={{
          height: "95%",
          borderRadius: `calc(${borderRadius} * 0.96)`,
        }}
      >
        {text}
      </div>
    </Component>
  );
}

export const MovingBorder = ({
  children,
  duration = 3000,
  rx,
  ry,
  ...otherProps
}: {
  children: React.ReactNode;
  duration?: number;
  rx?: string;
  ry?: string;
  [key: string]: any;
}) => {
  const pathRef = useRef<SVGRectElement | null>(null);
  const progress = useMotionValue<number>(0);

  useAnimationFrame((time) => {
    const length = pathRef.current?.getTotalLength();
    if (length) {
      const pxPerMillisecond = length / duration;
      progress.set((time * pxPerMillisecond) % length);
    }
  });

  const x = useTransform(progress, (val) =>
    pathRef.current?.getPointAtLength(val).x
  );
  const y = useTransform(progress, (val) =>
    pathRef.current?.getPointAtLength(val).y
  );

  const transform = useMotionTemplate`translateX(${x}px) translateY(${y}px) translateX(-50%) translateY(-50%)`;

  return (
    <>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="none"
        className="absolute h-full w-full"
        width="100%"
        height="100%"
        {...otherProps}
      >
        <rect
          fill="none"
          width="100%"
          height="100%"
          rx={rx}
          ry={ry}
          ref={pathRef}
        />
      </svg>
      <motion.div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          display: "inline-block",
          transform,
          zIndex: 0,
        }}
      >
        {children}
      </motion.div>
    </>
  );
};

// "use client";
// import React from "react";
// import {
//   motion,
//   useAnimationFrame,
//   useMotionTemplate,
//   useMotionValue,
//   useTransform,
// } from "motion/react";
// import { useRef } from "react";
// import { cn } from "@/lib/utils";

// export function Buttn({
//   borderRadius = "1.75rem",
//   children,
//   as: Component = "button",
//   containerClassName,
//   borderClassName,
//   duration,
//   className,
//   ...otherProps
// }: {
//   borderRadius?: string;
//   children: React.ReactNode;
//   as?: any;
//   containerClassName?: string;
//   borderClassName?: string;
//   duration?: number;
//   className?: string;
//   [key: string]: any;
// }) {
//   return (
//     <Component
//       className={cn(
//         "relative flex items-center justify-center h-16 w-40 overflow-hidden bg-transparent p-[1px] text-xl",
//         containerClassName,
//       )}
//       style={{
//         borderRadius: borderRadius,
        
//       }}
//       {...otherProps}
//     >
//       <div
//         className="absolute inset-0"
//         style={{ borderRadius: `calc(${borderRadius} * 0.96)` }}
//       >
//         <MovingBorder duration={duration} rx="30%" ry="30%">
//           <div
//             className={cn(
//               "h-20 w-20 bg-[radial-gradient(#0ea5e9_40%,transparent_60%)] opacity-[0.8]",
//               borderClassName,
//             )}
//           />
//         </MovingBorder>
//       </div>

//       <div
//         className={cn(
//           "relative flex h-full w-full items-center justify-center border border-slate-800 bg-slate-900/[0.8] text-sm text-white antialiased backdrop-blur-xl",
//           className,
//         )}
//         style={{
//           width:'97%',
//           height:'95%',
//           borderRadius: `calc(${borderRadius} * 0.96)`,
//         }}
//       >
        
//         {children}

        
//       </div>
//     </Component>
//   );
// }

// export const MovingBorder = ({
//   children,
//   duration = 3000,
//   rx,
//   ry,
//   ...otherProps
// }: {
//   children: React.ReactNode;
//   duration?: number;
//   rx?: string;
//   ry?: string;
//   [key: string]: any;
// }) => {
//   const pathRef = useRef<any>();
//   const progress = useMotionValue<number>(0);

//   useAnimationFrame((time) => {
//     const length = pathRef.current?.getTotalLength();
//     if (length) {
//       const pxPerMillisecond = length / duration;
//       progress.set((time * pxPerMillisecond) % length);
//     }
//   });

//   const x = useTransform(
//     progress,
//     (val) => pathRef.current?.getPointAtLength(val).x,
//   );
//   const y = useTransform(
//     progress,
//     (val) => pathRef.current?.getPointAtLength(val).y,
//   );

//   const transform = useMotionTemplate`translateX(${x}px) translateY(${y}px) translateX(-50%) translateY(-50%)`;

//   return (
//     <>
//       <svg
//         xmlns="http://www.w3.org/2000/svg"
//         preserveAspectRatio="none"
//         className="absolute h-full w-full"
//         width="100%"
//         height="100%"
//         {...otherProps}
//       >
//         <rect
//           fill="none"
//           width="100%"
//           height="100%"
//           rx={rx}
//           ry={ry}
//           ref={pathRef}
//         />
//       </svg>
//       <motion.div
//         style={{
//           position: "absolute",
//           top: 0,
//           left: 0,
//           display: "inline-block",
//           transform,
//           zIndex:0,
//           // width:'10px'
//         }}
//       >
//         {children}
//       </motion.div>
//     </>
//   );
// };
