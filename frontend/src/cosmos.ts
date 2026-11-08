// Fragment shader for the background: one camera move from low Earth orbit into a yellow-orange black hole.
// Z is depth (0 Earth, 1 event horizon), V is camera speed (stretches stars), X shifts the hole sideways for layout.
export const COSMOS_FRAG = `precision highp float;
uniform vec2 R;uniform float T;uniform vec2 M;uniform float E;uniform float D;uniform float Z;uniform float V;uniform float X;
float h21(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float n2(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h21(i),h21(i+vec2(1.,0.)),f.x),mix(h21(i+vec2(0.,1.)),h21(i+vec2(1.,1.)),f.x),f.y);}
float fbm(vec2 p){float a=.5,s=0.;for(int i=0;i<5;i++){s+=a*n2(p);p=p*2.03+7.1;a*=.5;}return s;}
float dust(vec2 p){vec2 g=p*70.;vec2 i=floor(g);float r=h21(i);vec2 o=vec2(h21(i+3.1),h21(i+9.7));float d=length(fract(g)-o);return smoothstep(.07,.0,d)*step(.93,r)*(.25+.5*h21(i+1.7));}
float spark(vec2 p){vec2 g=p*6.5;vec2 i=floor(g);float r=h21(i+.5);if(r<.7)return 0.;vec2 o=.2+.6*vec2(h21(i+3.1),h21(i+9.7));vec2 d=fract(g)-o;
float tw=.55+.45*sin(T*.8+r*40.);float core=exp(-dot(d,d)*1400.);
float cr=exp(-abs(d.x)*120.)*exp(-abs(d.y)*20.)+exp(-abs(d.y)*120.)*exp(-abs(d.x)*20.);return (core+cr*.6)*tw*(.4+.6*h21(i+1.7));}
vec3 ramp(float x){x=clamp(x,0.,1.);vec3 c=mix(vec3(.42,.07,0.),vec3(1.,.42,.02),smoothstep(0.,.42,x));c=mix(c,vec3(1.,.74,.16),smoothstep(.38,.72,x));return mix(c,vec3(1.,.93,.66),smoothstep(.8,1.,x));}
vec3 disk(vec2 q,float rs,float spin){float r=length(q);float a=atan(q.y,q.x);float inner=rs*1.5,outer=rs*5.;
float band=smoothstep(inner,inner*1.15,r)*smoothstep(outer,inner*1.7,r);
float s=a+T*spin*rs/max(r,.001)*2.;float sw=fbm(vec2(cos(s),sin(s))*2.6+r/rs*2.2);
float dop=.55+.45*cos(a-2.8);return ramp(band*(.12+1.0*sw)*(.42+dop*.72))*band*.92;}
void main(){
vec2 p=(gl_FragCoord.xy-.5*R)/R.y;float asp=R.x/R.y;vec2 par=(M-.5)*.012;
float s0=min(1.,asp/1.25);float sc=mix(s0,max(s0,.62),Z);float rs=.021*exp(Z*3.0)*sc;
vec2 C=vec2(X*(1.-smoothstep(.8,1.,Z)),mix(asp>1.15?.35:.26,0.,smoothstep(0.,.6,Z)))+par;vec2 d=p-C;float r=length(d);
float b=rs*rs*1.8/max(r,rs*.6);vec2 L=p-normalize(d)*b;
vec2 Ld=(L-C)/(1.+Z*1.4)+C;vec2 Ls=(L-C)/(1.+Z*.9)+C;
vec3 col=vec3(.002,.004,.012)+vec3(0.,.012,.05)*pow(fbm(Ld*1.6+T*.003),2.6)*1.4;
float sk=0.;for(int i=0;i<4;i++){sk+=spark(C+(Ls-C)*(1.+float(i)*V*.05)+par);}
col+=vec3(.85,.91,1.)*(dust(Ld+par*.5)*1.5+sk*.36*(1.+V*1.5));
float ca=cos(.1),sa=sin(.1);vec2 e=vec2(ca*d.x-sa*d.y,sa*d.x+ca*d.y);vec2 q=vec2(e.x,e.y/mix(.22,.36,Z));
float spin=.5+E*2.6;float hot=1.-.4*smoothstep(.72,1.,Z);vec3 dk=disk(q,rs,spin)*hot;float rn=r/rs;
vec3 arc=disk(normalize(e)*rs*(1.95+max(rn-1.12,0.)*1.9),rs,spin)*smoothstep(2.9,1.12,rn);
float shadow=smoothstep(rs*.97,rs*1.05,r);
col=col*shadow+arc*hot*shadow*(.82+E*.5);
col+=vec3(1.,.72,.3)*exp(-pow((r-rs*1.08)/(rs*.04),2.))*(.95+E*1.2);
col+=vec3(1.,.36,.03)*(.07+E*.08)*rs/max(r,rs)*1.7*shadow;col=mix(col,col+dk,e.y>0.?shadow:1.);
float ez=smoothstep(0.,.42,Z);float er=max(.98,asp*.62)*(1.+ez*.5);vec2 EC=vec2(0.,-.285-er-ez*1.25)-par*1.6;vec2 ed=(p-EC)/er;float ee=dot(ed,ed);
float warm=exp(-pow((p.x-C.x)/(.34*s0+.05),2.));float dl=(1.-sqrt(ee))*er;
if(ee<1.){vec3 n=vec3(ed,sqrt(1.-ee));float rim=exp(-dl*17.);
col=vec3(.002,.008,.024)+vec3(.01,.2,1.)*rim*1.25+vec3(.015,.08,.34)*exp(-dl*7.)*.42+vec3(.5,.2,.0)*warm*exp(-dl*60.)*.35;
vec2 sp=n.xy*5.5+vec2(T*.008,0.);float f=fbm(sp);float line=smoothstep(.016,.0,abs(f-.5));
float g2=fbm(sp*2.3+3.);float fine=smoothstep(.012,.0,abs(g2-.48));
col+=vec3(.42,.66,1.)*(line*.72+fine*.2)*(.12+exp(-dl*5.)*.8);
}else{float x=(sqrt(ee)-1.)*er;col+=(vec3(.05,.34,1.)*exp(-x*30.)*.8+vec3(.02,.1,.45)*exp(-x*9.)*.2)*mix(1.,shadow,.92)+vec3(1.,.45,.1)*warm*exp(-x*30.)*.45;}
vec2 v=p/vec2(asp,1.);col*=(1.-.32*dot(v,v))*(1.-.66*D);
vec3 o=clamp(pow(col,vec3(.9)),0.,1.);o=mix(vec3(dot(o,vec3(.3,.59,.11))),o,1.22);o=clamp(o,0.,1.);o=mix(o,o*o*(3.-2.*o),.6);gl_FragColor=vec4(o,1.);}`;

export const COSMOS_VERT = "attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}";
