import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { LockKeyhole, User } from "lucide-react"
import Image from "next/image"
import Link from "next/link"

export default function AdminLogin() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <header className="bg-[#003366] text-white py-4 px-6 shadow-md">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Image
              src="/placeholder.svg?height=40&width=40"
              width={40}
              height={40}
              alt="LVA Logo"
              className="rounded-full bg-white p-1"
            />
            <span className="font-semibold text-lg">Lehigh Valley Academy RCS</span>
          </div>
          <nav>
            <Link href="#" className="text-sm hover:underline">
              Contact
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center p-6">
        <Card className="w-full max-w-md shadow-lg border-t-4 border-t-[#003366]">
          <CardHeader className="space-y-1 text-center">
            <CardTitle className="text-2xl font-bold text-[#003366]">Admin Portal</CardTitle>
            <CardDescription>Enter your credentials to access the admin dashboard</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username" className="text-sm font-medium">
                Username
              </Label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400">
                  <User size={18} />
                </div>
                <Input id="username" placeholder="Enter your username" className="pl-10" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password" className="text-sm font-medium">
                  Password
                </Label>
                <Link href="#" className="text-xs text-[#003366] hover:underline">
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400">
                  <LockKeyhole size={18} />
                </div>
                <Input id="password" type="password" placeholder="••••••••" className="pl-10" />
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button className="w-full bg-[#003366] hover:bg-[#002244]">Sign In</Button>
          </CardFooter>
        </Card>
      </main>

      <footer className="bg-white py-4 px-6 border-t border-gray-200">
        <div className="container mx-auto text-center text-sm text-gray-600">
          <p>&copy; {new Date().getFullYear()} Lehigh Valley Academy Regional Charter School. All rights reserved.</p>
          <div className="mt-2 flex justify-center space-x-4">
            <Link href="#" className="hover:text-[#003366]">
              Privacy Policy
            </Link>
            <Link href="#" className="hover:text-[#003366]">
              Terms of Use
            </Link>
            <Link href="#" className="hover:text-[#003366]">
              Support
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}