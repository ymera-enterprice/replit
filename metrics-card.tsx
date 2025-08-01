import { Card, CardContent } from "@/components/ui/card";
import { Server, Zap, Users, Database, TrendingUp, TrendingDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricsCardProps {
  title: string;
  value: string;
  icon: "server" | "zap" | "users" | "database";
  trend?: {
    value: string;
    direction: "up" | "down";
  };
  subtitle?: string;
  color: "success" | "primary" | "secondary" | "warning";
}

const iconMap = {
  server: Server,
  zap: Zap,
  users: Users,
  database: Database,
};

const colorMap = {
  success: "bg-success/10 text-success",
  primary: "bg-primary/10 text-primary", 
  secondary: "bg-secondary/10 text-secondary",
  warning: "bg-warning/10 text-warning",
};

export function MetricsCard({ title, value, icon, trend, subtitle, color }: MetricsCardProps) {
  const Icon = iconMap[icon];

  return (
    <Card className="metrics-card">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold text-foreground">{value}</p>
          </div>
          <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center", colorMap[color])}>
            <Icon className="h-6 w-6" />
          </div>
        </div>
        {(trend || subtitle) && (
          <div className="mt-4 flex items-center text-sm">
            {trend && (
              <>
                <span className={cn(
                  "flex items-center",
                  trend.direction === "up" ? "text-success" : "text-success"
                )}>
                  {trend.direction === "up" ? (
                    <TrendingUp className="h-4 w-4 mr-1" />
                  ) : (
                    <TrendingDown className="h-4 w-4 mr-1" />
                  )}
                  {trend.value}
                </span>
                {subtitle && <span className="text-muted-foreground ml-2">{subtitle}</span>}
              </>
            )}
            {!trend && subtitle && (
              <span className={cn(
                color === "warning" ? "text-warning" : "text-muted-foreground"
              )}>
                {subtitle}
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
