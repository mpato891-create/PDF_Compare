import { Button } from "@/components/ui/button";

const App = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4">
      <Button>Click me</Button>
      <Button variant="outline">Outline</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="destructive">Delete</Button>
      <Button variant="ghost">Ghost</Button>
      <Button size="sm">Small</Button>
      <Button size="lg">Large</Button>
    </div>
  );
};

export default App;
